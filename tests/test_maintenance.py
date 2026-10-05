"""The background worker that embeds what a write left pending.

A write commits the statement and then hands it to this worker, so the record never
waits on the model. That trade only holds if a worker that cannot do its job keeps
waiting for a reason to try again, because the alternative is a process whose meaning
side is off for the rest of its life with a traceback nobody reads.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from memory_rag.maintenance import RETRY_SECONDS, EmbeddingWorker, PendingUnit
from memory_rag.models import ModelError


class _Unavailable(FakeEmbedder):
    """An embedder that is never available, and counts how often it was asked."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def embed_documents(self, texts):
        self.attempts += 1
        raise ModelError("no embedding model is available here")


@pytest.fixture(autouse=True)
def _short_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests to a second each without waiting out the product's interval."""

    monkeypatch.setattr("memory_rag.maintenance.RETRY_SECONDS", 0.2)


def test_a_model_that_cannot_be_loaded_leaves_the_statement_pending(
    tmp_path: Path,
) -> None:
    """A write never fails because of the lookup layer, and never succeeds either."""

    embedder = _Unavailable()
    worker = EmbeddingWorker(embedder, tmp_path / "scope")
    try:
        worker.submit(PendingUnit("a-key", "a statement", "RULE", "0"))
        assert worker.drain(1.0) is False
        assert worker.pending == 1

        # A later write is the retry, so the model is asked again rather than waiting.
        worker.submit(PendingUnit("another-key", "another statement", "RULE", "0"))
        assert worker.pending == 2
    finally:
        worker.stop()


def test_a_run_of_failures_is_said_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A condition is not a per-attempt event.

    A model that cannot be fetched is retried on an interval for as long as the process
    runs, so a line per attempt turns something a person has to act on into noise they
    learn to scroll past. A new write is a new run, because the new write is what
    somebody is waiting on.
    """

    caplog.set_level(logging.WARNING, logger="memory_rag.maintenance")
    worker = EmbeddingWorker(_Unavailable(), tmp_path / "scope")
    try:
        worker.submit(PendingUnit("a-key", "a statement", "RULE", "0"))
        time.sleep(1.0)
        first = [
            record for record in caplog.records if "stays pending" in record.message
        ]
        assert len(first) == 1, [record.message for record in caplog.records]
        assert "a-key"[:12] in first[0].message

        caplog.clear()
        worker.submit(PendingUnit("a-key", "a statement", "RULE", "0"))
        time.sleep(1.0)
        again = [
            record for record in caplog.records if "stays pending" in record.message
        ]
        assert len(again) == 1, "a new write is a new run of failures"
    finally:
        worker.stop()


def test_a_missing_model_is_retried_rather_than_spun_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The wait is what keeps a failure costing one attempt per interval."""

    caplog.set_level(logging.WARNING, logger="memory_rag.maintenance")
    embedder = _Unavailable()
    worker = EmbeddingWorker(embedder, tmp_path / "scope")
    try:
        worker.submit(PendingUnit("a-key", "a statement", "RULE", "0"))
        time.sleep(1.0)
    finally:
        worker.stop()
    assert 1 < embedder.attempts < 25, f"the worker tried {embedder.attempts} times"
    assert RETRY_SECONDS == 5.0, "the product's own interval moved"


def test_a_storage_failure_leaves_the_unit_pending_and_the_worker_running(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure of the lookup layer must not take the queue with it.

    The unit was taken out of the queue and the thread died with it, so the next write
    was queued for a thread that no longer existed: the record kept every statement and
    the meaning side of that process was off for the rest of its life. The failure is
    reported, the statement stays pending, and the worker keeps waiting for a reason to
    try again.
    """

    caplog.set_level(logging.WARNING, logger="memory_rag.maintenance")
    scope = tmp_path / "not-a-directory"
    scope.write_text("a file where a scope directory belongs")
    worker = EmbeddingWorker(FakeEmbedder(), scope)
    try:
        worker.submit(PendingUnit("a-key", "a statement", "NOTE", "0"))
        assert worker.drain(1.0) is False

        assert worker.pending == 1
        worker.submit(PendingUnit("another-key", "another statement", "NOTE", "0"))
        assert worker.pending == 2
        said = [
            record for record in caplog.records if "stays pending" in record.message
        ]
        assert len(said) == 1, "one run of failures is said once"
    finally:
        worker.stop()


def test_a_storage_failure_is_reported_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The reason has to name the failure, or the line is not actionable."""

    caplog.set_level(logging.WARNING, logger="memory_rag.maintenance")
    scope = tmp_path / "scope"
    scope.mkdir()
    worker = EmbeddingWorker(FakeEmbedder(), scope)
    monkeypatch.setattr(
        "memory_rag.maintenance.VectorStore",
        _store_that_raises(sqlite3.OperationalError("disk I/O error")),
    )
    try:
        worker.submit(PendingUnit("a-key", "a statement", "NOTE", "0"))
        assert worker.drain(1.0) is False
        assert worker.pending == 1
    finally:
        worker.stop()
    said = [
        record.getMessage()
        for record in caplog.records
        if "stays pending" in record.getMessage()
    ]
    assert said and "disk I/O error" in said[0]


def test_a_worker_that_recovered_does_not_say_anything_on_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A success ends the run, so the next failure is a new thing to say."""

    caplog.set_level(logging.WARNING, logger="memory_rag.maintenance")
    scope = tmp_path / "scope"
    scope.mkdir()
    healthy: list[bool] = [True]
    real = __import__("memory_rag.vectors", fromlist=["VectorStore"]).VectorStore

    def _sometimes(store_directory, model, dimension):
        store = real(store_directory, model, dimension)
        if not healthy[0]:
            monkeypatch.setattr(
                store, "put", _refusing(sqlite3.OperationalError("disk I/O error"))
            )
        return store

    monkeypatch.setattr("memory_rag.maintenance.VectorStore", _sometimes)
    worker = EmbeddingWorker(FakeEmbedder(), scope)
    try:
        worker.submit(PendingUnit("a-key", "a statement", "NOTE", "0"))
        assert worker.drain(5.0) is True

        healthy[0] = False
        caplog.clear()
        worker.submit(PendingUnit("b-key", "another statement", "NOTE", "0"))
        assert worker.drain(1.0) is False
        said = [
            record.getMessage()
            for record in caplog.records
            if "stays pending" in record.getMessage()
        ]
        assert len(said) == 1
    finally:
        worker.stop()


def _store_that_raises(error: Exception):
    """Return a stand-in vector store whose every call raises."""

    def factory(*_args, **_kwargs):
        raise error

    return factory


def _refusing(error: Exception):
    def refuse(*_args, **_kwargs):
        raise error

    return refuse
