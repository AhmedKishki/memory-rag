"""Keeping the derived state in step with the record, on the write side.

Three rules, in the order they happen:

1. **Durability first.** The statement is a row in the record before anything here
   runs, so nothing in this module can lose a memory.
2. **No work proportional to the memory.** What is asked about is which statements
   the vector table is missing, which is a set difference over keys rather than a
   pass over the memory.
3. **A write never fails because of the lookup layer.** If the model is
   unavailable the units are left pending, the write says so, and the next write
   or an explicit reindex picks them up.

This is also where the deliberate trade is made explicit. Maintaining the index
here rather than on the read path is what keeps a lookup O(1) in the number of
statements; the cost is that the vector side trails a write whose model was not
available, and a read reports that it is behind rather than pretending otherwise.
:func:`reindex` exists for that.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .index import RENDERED_FILENAME, MemoryIndex
from .models import Embedder, ModelError
from .store import StoreError
from .vectors import VectorStore

LOGGER = logging.getLogger(__name__)

#: How long the worker waits for work when nothing is wrong. It bounds how long a
#: stopped or idle worker notices `stop`, and nothing else.
POLL_SECONDS = 0.25

#: How long the worker waits before retrying a model it could not load. A model
#: that cannot be fetched stays unavailable for minutes, so the wait is long
#: enough to cost nothing and short enough that a model fetched by another process
#: is picked up within the next write's lifetime. `submit` clears it, which is what
#: makes the next write the retry rather than the timer.
RETRY_SECONDS = 5.0

__all__ = [
    "EmbeddingWorker",
    "PendingUnit",
    "embed_pending",
    "reindex",
    "warm_models",
]

#: How many pending units one drain pass will embed. Bounded so a maintenance
#: call cannot run for a minute, and so a reindex converges over several calls.
DEFAULT_BATCH = 64


@dataclass(frozen=True, slots=True)
class PendingUnit:
    """One statement waiting for a vector: its text, its type, and where it sits."""

    key: str
    text: str
    kind: str
    stamp: str


class EmbeddingWorker:
    """Produces vectors off the caller's thread, in small batches.

    One queue and one thread for the whole process, because a memory has one
    scope at a time here and a second worker would fight the first over the same
    SQLite file. A unit is embedded once: the queue is drained, the model is
    called with a batch, and the vectors are written. If the model or the storage
    is not available the batch is put back and the worker waits before trying
    again, so a missing model costs one wakeup per interval rather than a core, and
    the next write or an explicit reindex clears the wait and retries at once.
    """

    def __init__(self, embedder: Embedder, scope_directory: Path) -> None:
        self._embedder = embedder
        self._scope_directory = scope_directory
        self._queue: queue.Queue[PendingUnit | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._idle = threading.Event()
        # Set by `stop`, and read by the retry below. Without it a worker whose
        # model is missing never reaches the `None` at the end of its queue: it
        # puts the unit back and takes it out again, so `stop` waits out its whole
        # timeout and leaves a thread spinning on the failure.
        self._stopping = False
        #: When the worker may try the model again, as a monotonic time. This is
        #: what a retry waits on: a failed embed puts the unit back and returns,
        #: and the loop below takes it out again only once this has passed. Without
        #: it the thread re-tried a missing model as fast as it could fail, which
        #: measured at over a million attempts in two seconds.
        self._retry_after = 0.0
        #: Whether the worker is in a run of failures, so a condition is reported
        #: once rather than once per attempt. A write or a success ends the run.
        self._failing = False
        self._idle.set()

    @property
    def pending(self) -> int:
        """Return how many units are queued right now."""

        return self._queue.qsize()

    def start(self) -> None:
        """Start the worker thread, if it is not already running."""

        with self._lock:
            if self._thread is not None:
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run, name="memory-embedding", daemon=True
            )
            self._thread.start()

    def submit(self, unit: PendingUnit) -> None:
        """Queue one unit for a vector, and wake the worker.

        A write is a reason to try the model now: whatever wait a previous failure
        put in place is cleared, so a model that has become available again is
        picked up by the next write rather than at the end of a backoff.
        """

        self.start()
        with self._lock:
            self._retry_after = 0.0
            self._failing = False
        self._idle.clear()
        self._queue.put(unit)

    def drain(self, timeout: float = 5.0) -> bool:
        """Wait for the queue to empty, and report whether it did.

        A test and the reindex command need the work to be *done*, not merely
        queued, so this waits for the worker rather than for the queue.
        """

        return self._idle.wait(timeout)

    def stop(self) -> None:
        """Ask the worker to finish and wait for it.

        Whatever is still queued is dropped, because the thread is gone and
        nothing else reads that queue. Dropping a pending unit loses nothing: the
        statement is in the record, and the next worker derives the missing
        vectors from it rather than from this queue.
        """

        if self._thread is None:
            return
        self._stopping = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        self._thread = None
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        while True:
            wait = self._retry_wait()
            if wait:
                # The unit is already back on the queue, so waiting on `get` would
                # hand it straight back out: the wait has to happen before the take,
                # or a missing model is retried as fast as it can fail. Sliced, so a
                # `stop` is still heard within a poll.
                if self._stopping:
                    self._idle.set()
                    return
                time.sleep(min(wait, POLL_SECONDS))
                continue
            try:
                item = self._queue.get(timeout=POLL_SECONDS)
            except queue.Empty:
                if self._queue.empty():
                    self._idle.set()
                continue
            if item is None:
                self._idle.set()
                return
            try:
                self._embed_one(item)
            except (OSError, sqlite3.Error, StoreError) as error:
                # A worker that dies takes the queue with it: the unit is gone from
                # the queue, the thread is gone with it, and every later write is
                # queued for a thread that no longer exists — one storage error, and
                # the meaning side of this process is off for good. So a failure of
                # the lookup layer is handled as a missing model is: the unit goes
                # back, the retry waits, and the read discloses what is pending.
                self._requeue(item, error)
            if self._queue.empty():
                self._idle.set()

    def _retry_wait(self) -> float:
        """Return how long is left of a retry wait, or zero when there is none."""

        with self._lock:
            return max(self._retry_after - time.monotonic(), 0.0)

    def _requeue(self, unit: PendingUnit, reason: Exception | None = None) -> None:
        """Put a unit back, wait before the next attempt, and say what failed.

        The unit stays queued so the statement is picked up by the next write or an
        explicit reindex, and the wait is what keeps a failure costing one attempt
        per interval. The reason is reported once per run of failures rather than
        once per attempt, so a model or a file that is simply not there says so in
        the app's log without repeating itself all day. A read says the same thing
        in `units_pending`.
        """

        if self._stopping:
            self._idle.set()
            return
        with self._lock:
            first = not self._failing
            self._failing = True
        if reason is not None and first:
            LOGGER.warning(
                "memory-embedding: %s was not embedded (%s); it stays pending and is "
                "retried every %ss",
                unit.key[:12],
                reason,
                f"{RETRY_SECONDS:.0f}",
            )
        self._queue.put(unit)
        self._idle.clear()
        with self._lock:
            self._retry_after = time.monotonic() + RETRY_SECONDS

    def _settled(self) -> None:
        """End a run of failures, because this unit is dealt with however it happened."""

        with self._lock:
            self._retry_after = 0.0
            self._failing = False

    def _embed_one(self, unit: PendingUnit) -> None:
        try:
            vectors = self._embedder.embed_documents([unit.text])
        except ModelError as error:
            # The model is not available. Put the unit back, wait before the next
            # attempt, and let a later write or a reindex retry it — unless the
            # worker is stopping, because a unit put back behind the `None` at the
            # end of the queue is one the thread would take out again and again,
            # and `stop` would wait out its timeout and leave it spinning.
            self._requeue(unit, error)
            return
        if not vectors:
            self._settled()
            return
        identity = self._embedder.identity
        with VectorStore(
            self._scope_directory, identity.name, identity.dimension
        ) as store:
            try:
                store.put(
                    unit.key,
                    vectors[0],
                    stamp=unit.stamp,
                    kind=unit.kind,
                    expected_text=unit.text,
                )
            except ValueError:
                # Another model's file: leave it for reindex rather than mixing.
                self._settled()
                return
        self._settled()


def _pending(index: MemoryIndex, key: str) -> PendingUnit:
    """Return the statement a key names, as the worker needs to embed it."""

    found = index.units_by_key([key]).get(key)
    if found is None:
        return PendingUnit(key, "", "ITEM", "0")
    return PendingUnit(key, str(found["text"]), str(found["kind"]), str(found["stamp"]))


def embed_pending(
    scope_directory: Path,
    embedder: Embedder,
    worker: EmbeddingWorker | None,
) -> int:
    """Queue the statements of a scope that have no vector, and return how many.

    The record is the table, so there is nothing to sync first and nothing to
    narrow: what is asked about is which statements the vector table is missing,
    which is a set difference over keys and not a pass over the memory.

    The statements are read through the one connection that is open for this
    scope, so a scope with a thousand pending units is opened once rather than
    once per unit.
    """

    identity = embedder.identity
    with MemoryIndex(scope_directory) as index:
        units = index.unit_keys()
        with VectorStore(scope_directory, identity.name, identity.dimension) as store:
            have = store.has(units)
        missing = units - have
        if worker is None:
            return len(missing)
        for key in missing:
            worker.submit(_pending(index, key))
        return len(missing)


def reindex(
    scope_directories: Sequence[Path],
    embedder: Embedder | None = None,
    *,
    worker: EmbeddingWorker | None = None,
    batch: int = DEFAULT_BATCH,
    retire_document: bool = False,
) -> dict[str, Any]:
    """Settle one or more scopes: recover, export, and embed what is missing.

    This is the answer to "the file and the database disagree". A record with
    nothing in it adopts the document, which is how a memory written by an older
    version is recovered; a document that does not match the record is rewritten
    from it; and the statements with no vector are embedded, which is the one thing
    here that costs a model.
    """

    report: dict[str, Any] = {"scopes": {}}
    # A worker writes vectors into the file of the scope it was built for, so it
    # can only serve that one. Rebuilding several scopes therefore embeds here,
    # in the process that is already doing maintenance work, rather than handing
    # a second scope's statements to the first scope's writer.
    queue = worker if len(scope_directories) == 1 else None
    for scope_directory in scope_directories:
        identity = embedder.identity if embedder is not None else None
        with MemoryIndex(scope_directory) as index:
            adopted = index.adopt_document_if_empty()
            retired = index.retire_superseded(render=retire_document)
            if retired and RENDERED_FILENAME in retired:
                # Retiring the rendering removes the file, and the record it was written
                # from is what a person opens to read a memory, so it is written again.
                store_rendered = str(index.write_render())
            else:
                store_rendered = ""
            units = index.unit_keys()
            store_report: dict[str, Any] = {
                "adopted_from_document": adopted,
                "superseded_removed": retired,
                "rendered": store_rendered,
                "units": len(units),
            }
            missing = set()
            if identity is not None:
                with VectorStore(
                    scope_directory, identity.name, identity.dimension
                ) as store:
                    if store.model_is_foreign():
                        store_report["vectors"] = "foreign model; rebuilt"
                        with MemoryIndex(scope_directory) as scope_file:
                            scope_file.drop_vectors()
                        with VectorStore(
                            scope_directory, identity.name, identity.dimension
                        ) as fresh:
                            missing = units - fresh.has(units)
                    else:
                        missing = units - store.has(units)
                    store_report["vectors_held"] = store.count()
            store_report["units_pending"] = len(missing)
            # A vector whose statement the record no longer holds is about nothing
            # that exists. A memory recovered from an export is the case that makes
            # them: the words come back and the types do not, so every recovered
            # statement is a new one and the old vectors are orphans.
            store_report["vectors_collected"] = index.collect_vectors()
            if embedder is not None and missing:
                rows = index.units_by_key(missing)
                if queue is not None:
                    for key in rows:
                        queue.submit(_pending(index, key))
                else:
                    # Several scopes, or no worker: embed here, in the process that
                    # is already doing maintenance work and has no caller waiting.
                    with VectorStore(
                        scope_directory, identity.name, identity.dimension
                    ) as store:
                        vectors = embedder.embed_documents(
                            [row["text"] for row in rows.values()]
                        )
                        for (key, row), vector in zip(
                            rows.items(), vectors, strict=True
                        ):
                            store.put(
                                key,
                                vector,
                                stamp=str(row["stamp"]),
                                kind=str(row["kind"]),
                            )
        report["scopes"][str(scope_directory)] = store_report
    return report


def warm_models(
    embedder: Embedder,
    reranker: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Load the models now, so a later read finds them resident.

    Loading is seconds, and a read that paid for it would stall a caller. The
    write side has no deadline, so it is the side that loads.
    """

    report: dict[str, Any] = {}
    try:
        embedder.warm()
        report["embedding_model"] = str(embedder.identity)
        report["embedding_loaded"] = True
    except ModelError as error:
        report["embedding_model"] = str(embedder.identity)
        report["embedding_loaded"] = False
        report["embedding_error"] = str(error)
    if reranker is not None:
        try:
            reranker()
            report["reranker_loaded"] = True
        except ModelError as error:
            report["reranker_loaded"] = False
            report["reranker_error"] = str(error)
    return report
