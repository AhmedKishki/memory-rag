"""What a read is allowed to do, in one object.

Retrieval has levers — a model, a floor, two weights, a depth, a reranker — and
every one of them is a decision someone could argue about. They live here, in one
frozen object read from one place, so the later measurement can sweep them
without a single call site changing, and so a reader can see the whole shape of
a lookup in one screen.

The defaults are the collection's measured ones rather than invented numbers:
weighted reciprocal rank fusion with ``rrf_k = 60``, BM25 weighted 1.25 against
dense 1.0, a cosine floor of 0.72, and a relative margin of 0.10 around the
query's best candidate. They are the sibling server's, measured on a document
corpus rather than on memory units, and what decides whether they hold here is a
measurement this package has not yet made. Reranking is on, because a memory's
units are one line each, which is where a bi-encoder is weakest, and a
cross-encoder reads the pair rather than either side alone; its cost is bounded
by ``rerank_depth`` against the read's ceiling. That it is more *accurate* here
is unmeasured, and nothing in this package claims it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .index import MemoryIndex
from .maintenance import EmbeddingWorker, embed_pending, warm_models
from .models import DEFAULT_EMBEDDING_MODEL, Embedder, ModelError, Reranker
from .read import Answer, answer_read, merge_answers
from .store import (
    HANDOFF_KIND,
    Statement,
    StoreError,
    admission_failure,
    statement_kind,
)
from .vectors import VectorStore

if TYPE_CHECKING:
    from .settings import EffectiveSettings

__all__ = ["Retrieval", "RetrievalSettings"]


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """One read's policy: what to embed, how to match, what to admit, what to return."""

    #: The local model a scope's vectors are built with. Changing it invalidates
    #: them, by fingerprint, rather than mixing two spaces.
    embedding_model: str

    #: A dense candidate below this cosine is dropped, so a query with nothing to
    #: do with the memory abstains instead of receiving the nearest thing.
    cosine_floor: float = 0.72

    #: A candidate below the floor is admitted only when the query's own best
    #: candidate cleared the floor and this one is within this much of it.
    relative_margin: float = 0.10

    #: Reciprocal rank fusion: each side contributes ``weight / (rrf_k + rank)``.
    bm25_weight: float = 1.25
    dense_weight: float = 1.0
    rrf_k: int = 60

    #: What fraction of its own score a statement is worth for being the newest
    #: of the candidates that matched about as well. A statement is dated when it
    #: is recorded, and the newest one among equals gets the whole bonus, the next
    #: half of it, and so on. At 0.1 a materially better match still comes first
    #: and a match that is close to it does not; this is a knob and not a
    #: guarantee, so a caller who sets it to 1.0 is asking for the newest answer
    #: whatever it says. Zero turns it off, which is what a memory that must be
    #: reproducible from its own scores wants.
    recency_bonus: float = 0.1

    #: How alike two statements must be before a read treats them as one thing and
    #: shows only the better-ranked of them. Two statements of different words can
    #: still be one statement, and a memory that answers a question twice is a
    #: memory nobody can be sure of. This number decides that part only: words that
    #: are the same words are one statement whatever it says, and a number no
    #: cosine can reach decides that nothing is close enough rather than turning
    #: the check off.
    #:
    #: 0.85, not 0.99. Measured over every pair in a live memory of 23 statements:
    #: 0.99 caught none of the 253 pairs, and 0.85 caught the five that restate
    #: one fact in other words, which is the case this exists for. The nearest pair
    #: in that memory is 0.903 and its median is 0.716, so a threshold near one
    #: sees nothing at all and this one still leaves the distinctly-worded pairs
    #: alone. It is a knob: a memory whose statements are near neighbours by nature
    #: wants it lower, and one whose statements are short wants it higher.
    duplicate_cosine: float = 0.85

    #: How many candidates each side contributes before fusion. This is an
    #: internal depth, not the answer: the answer is exactly ``limit`` units, and
    #: every unit is text the caller has to read.
    pool_depth: int = 50

    #: The local cross-encoder that orders the fused candidates. Every read is
    #: reranked: a memory's units are one line each, where a bi-encoder's vector
    #: is weakest, and a cross-encoder reads the pair rather than either side
    #: alone. Measured on this machine at 54 ms for ten candidates, which is what
    #: sets :attr:`rerank_depth`. No document here may claim it is more accurate:
    #: that is unmeasured, and the first thing to measure. It is not optional, so
    #: there is no setting that switches it off and no empty value for one.
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    #: How many fused candidates the reranker sees, so its cost is bounded even
    #: if the pool grows. Ten, because the read's own ceiling is 250 ms and the
    #: cross-encoder costs about 5 ms a candidate on this machine. It bounds the
    #: cost and never the ranking: at 1 every read is reranked on its best
    #: candidate, which changes the answer and not only its cost.
    rerank_depth: int = 10

    #: What a warm read may cost, asserted by a test so a later change cannot
    #: quietly make a blocking lookup slow. 500 ms, not 250: the cross-encoder
    #: costs about 5 ms a candidate in a tight loop but roughly 190 ms per read
    #: through the library, and the first inference in a process is a model's own
    #: cold start, which this bound does not cover — the test excludes the first
    #: read for the same reason. A read that finds the model unloaded is outside
    #: this bound and says so in its answer.
    #:
    #: The suite asserts this against a fake reranker, which costs nothing, so it
    #: guards the fusion and the scan and not the cross-encoder. The real figure
    #: was measured on this machine, out of band: 54 ms for ten candidates scored
    #: in a loop, about 210 ms for a warm read end to end, and about 1.5 s for
    #: the first reranked read in a process.
    read_ceiling_ms: float = 500.0

    @classmethod
    def from_settings(cls, settings: EffectiveSettings | None) -> RetrievalSettings:
        """Return the policy the settings ask for, or the defaults when there are none.

        The settings are the only place these numbers are written down, so a
        ``config.toml`` naming one of them is the way to change it, and the
        defaults here are the same numbers the packaged file declares.
        """

        if settings is None:
            return cls(embedding_model=DEFAULT_EMBEDDING_MODEL)
        if not str(settings.reranker_model or "").strip():
            # Every read is reranked, so a policy with no cross-encoder is not a
            # policy this server can serve. The layer machinery lets an empty
            # string through for every string setting, so the one value that would
            # switch the reranker off is refused here, by name, before serving.
            raise ModelError(
                "dense.reranker_model must name one of the pinned rerankers, and "
                "there is no empty value: every read is reranked. Name one in "
                "config.toml, or leave the setting out to take the packaged default."
            )
        return cls(
            embedding_model=settings.embedding_model,
            cosine_floor=settings.cosine_floor,
            relative_margin=settings.relative_margin,
            bm25_weight=settings.bm25_weight,
            dense_weight=settings.dense_weight,
            rrf_k=settings.rrf_k,
            recency_bonus=settings.recency_bonus,
            duplicate_cosine=settings.duplicate_cosine,
            pool_depth=settings.pool_depth,
            reranker_model=settings.reranker_model,
            rerank_depth=settings.rerank_depth,
        )

    def describe(self) -> dict[str, object]:
        """Return this policy, for a diagnostic surface and for a test."""

        return {
            "embedding_model": self.embedding_model,
            "cosine_floor": self.cosine_floor,
            "relative_margin": self.relative_margin,
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "rrf_k": self.rrf_k,
            "recency_bonus": self.recency_bonus,
            "pool_depth": self.pool_depth,
            "reranker_model": self.reranker_model,
            "rerank_depth": self.rerank_depth,
            "read_ceiling_ms": self.read_ceiling_ms,
        }


@dataclass(frozen=True, slots=True)
class Found:
    """One statement that matched an exact text, and the memory it was found in."""

    statement: Statement
    scope: str
    directory: Path


class Retrieval:
    """One process's retrieval: its models, its policy, and its per-scope workers.

    This is the single object the tools and the browser view share, so a page
    write and a tool write leave the same derived state behind them, and so a
    model is loaded once for the process rather than once per caller. Nothing
    here holds a scope's memory in memory: it holds a path, and the indexes are
    opened and closed around each operation.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        policy: RetrievalSettings,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.policy = policy
        self.reranker = reranker
        self._workers: dict[str, EmbeddingWorker] = {}

    def worker_for(self, directory: Path) -> EmbeddingWorker:
        """Return this scope's worker, creating it once per process."""

        key = str(directory)
        if key not in self._workers:
            self._workers[key] = EmbeddingWorker(self.embedder, directory)
        return self._workers[key]

    def close(self) -> None:
        """Stop this process's background workers."""

        for worker in self._workers.values():
            worker.stop()
        self._workers.clear()

    def warm(self) -> dict[str, Any]:
        """Load the models now, so a later read finds them resident."""

        return warm_models(
            self.embedder, self.reranker.warm if self.reranker is not None else None
        )

    def record(
        self,
        *,
        content: str,
        directory: Path,
        kind: str | None = None,
        replace_kind: str | None = None,
    ) -> dict[str, Any]:
        """Record one statement in one memory, and say what was filed where.

        The statement is a row: its text, the kind the caller named, its place at
        the top of the document, and the time it was recorded. The document is
        then written out from the record, because a memory nobody can open is not
        one anybody can check.

        Two things are refused before the row is written, and both are refused on
        the statement's own words rather than on anything derived, so the same text
        is judged the same way every time. A statement carrying a date or a clock
        time reports a moment instead of saying what holds, which is
        :func:`store.admission_failure`'s one rule; and the reserved ``HANDOFF``
        kind is refused here, because a handoff is replaced by the next one and an
        ordinary record filed under it would be removed by the next session's
        handoff without ever being seen.

        Nothing is refused for being similar to what is already here. A memory that
        refuses a write is a memory a caller cannot write to, and deciding that on
        a write means deciding it with whatever the vectors happen to say that
        moment — a write does not wait for a vector, so the same statement would be
        judged differently depending on what had been embedded. The repetition is
        resolved where it can be judged on the whole answer instead, in the read,
        and there it collapses rather than refuses.
        """

        statement = str(content or "").strip()
        if not statement:
            raise ModelError("content must not be empty.")
        failure = admission_failure(statement)
        if failure is not None:
            raise ModelError(failure)
        try:
            label = statement_kind(kind)
        except StoreError as error:
            raise ModelError(str(error)) from error
        if label == HANDOFF_KIND and replace_kind != HANDOFF_KIND:
            raise ModelError(
                "HANDOFF is reserved for record_handoff, which holds one and replaces "
                "it with the next session's, so a statement filed under it here would "
                "be removed by the next handoff. Use record_handoff, or file this "
                "statement under a kind of your own."
            )
        with MemoryIndex(directory) as index:
            _key, replaced = index.insert(statement, label, replace_kind=replace_kind)
            # A statement whose words were reworded is a new identity, and the
            # vector of the old one describes text the record no longer holds. The
            # write path is where that happens, so this is where they are dropped,
            # and the settle that follows re-embeds what is left.
            stale = index.collect_vectors()
        embed_pending(directory, self.embedder, self.worker_for(directory))
        # What a write answers with is what it filed: the statement, the kind it
        # was filed under, and anything it removed to file it. The rest is
        # bookkeeping — the digest that identifies the statement, the directory it
        # went to, the model, the queue — and a caller can act on none of it.
        recorded: dict[str, Any] = {
            "status": "recorded",
            "kind": label,
            "text": statement,
        }
        if replaced:
            recorded["replaced"] = replaced
        if stale:
            recorded["vectors_removed"] = stale
        return recorded

    def handoff(self, directory: Path, content: str) -> dict[str, Any]:
        """Put this session's handoff at the top, replacing the last one.

        A handoff is one statement filed under the reserved kind `HANDOFF`. The
        previous one is removed by that kind, so a handoff is a standing single
        entry rather than a list of them, and the caller never has to remember
        that an older one was there. The count of what it replaced is returned,
        because a replaced statement is still data the caller may want to know it
        lost.
        """

        return self.record(
            directory=directory,
            content=content,
            kind=HANDOFF_KIND,
            replace_kind=HANDOFF_KIND,
        )

    def maintain(self, directory: Path) -> dict[str, Any]:
        """Bring the document in line with the record, and say whether it was stale.

        A record with nothing in it adopts a document left by an older version,
        which is how such a memory is recovered, and anything the document still
        held that the record lacks is imported and then removed. A rendering the
        record agrees with is left alone: it is what somebody asked for.
        """

        with MemoryIndex(directory) as index:
            adopted = index.adopt_document_if_empty()
            retired = index.retire_superseded()
            stale = index.collect_vectors()
        queued = embed_pending(directory, self.embedder, self.worker_for(directory))
        return {
            "adopted": adopted,
            "superseded_removed": retired,
            "vectors_removed": stale,
            "units_queued": queued,
        }

    def answer(
        self,
        *,
        scope: str,
        directory: Path,
        query: str,
        limit: int,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Answer one read of one memory, in words and in meaning."""

        return self.read(
            scope=scope, directory=directory, query=query, limit=limit, kind=kind
        ).payload()

    def read(
        self,
        *,
        scope: str,
        directory: Path,
        query: str,
        limit: int,
        kind: str | None = None,
    ) -> Answer:
        """Answer one read of one memory, and return the object it built."""

        return answer_read(
            scope=scope,
            directory=directory,
            query=query,
            limit=limit,
            kind=kind,
            embedder=self.embedder,
            reranker=self.reranker,
            policy=self.policy,
        )

    def answer_both(
        self,
        *,
        directories: Mapping[str, Path],
        query: str,
        limit: int,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Answer one read from every memory, ranked together.

        A caller asking what is remembered is asking one question, and the answer
        is the best statement from either memory rather than the best from each in
        turn. The scopes are searched in the order given and the results merged by
        score, so a statement is placed by how well it matched.
        """

        answers = [
            self.read(
                scope=scope,
                directory=directory,
                query=query,
                limit=limit,
                kind=kind,
            )
            for scope, directory in directories.items()
        ]
        merged = merge_answers(
            answers,
            limit=limit,
            duplicate_cosine=self.policy.duplicate_cosine,
        )
        payload = merged.payload()
        return payload

    def forget(
        self,
        *,
        text: str,
        directories: Mapping[str, Path],
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Remove the one statement this text is exactly, from one memory or either.

        The match is on the words and never on meaning, across every memory named:
        a statement the user has replaced is often worded the same way in one
        place and not the other, and a caller who cannot say which memory meant it
        should not have to guess. Two matches is a refusal, whichever memory they
        are in.
        """

        if not str(text or "").strip():
            raise ModelError(
                "text must not be empty: forgetting takes the exact text of the one "
                "statement to remove, so recall it first if you do not have it"
            )
        searched = (
            {scope: directories[scope]} if scope in directories else dict(directories)
        )
        found: list[Found] = []
        for name, directory in searched.items():
            with MemoryIndex(directory) as index:
                found.extend(
                    Found(statement, name, directory)
                    for statement in index.matches(text)
                )
        if not found:
            raise ModelError(
                "nothing matched exactly, so nothing was forgotten. This server "
                "forgets on an exact match only: recall the statement first, then "
                "forget the text the recall returned."
            )
        if len(found) > 1:
            # Each match is labelled with both, because the advice that follows
            # depends on which of the two distinguishes them.
            listed = "; ".join(
                f"[{item.scope} {item.statement.kind}] {item.statement.text}"
                for item in found[:5]
            )
            raise ModelError(
                f"{len(found)} statements match that text exactly, so none were "
                f"forgotten: {listed}. Pass a scope if they are in different "
                "memories, or forget a statement whose text is unique."
            )
        removed = found[0]
        with MemoryIndex(removed.directory) as index:
            index.drop(removed.statement.key)
        identity = self.embedder.identity
        with VectorStore(removed.directory, identity.name, identity.dimension) as store:
            store.forget([removed.statement.key])
        # What a forget answers with is what it removed, and how often the memory
        # had been handing it back: a caller that kept a count has to be able to
        # see what that count was. The keys and the directory are bookkeeping.
        forgotten: dict[str, Any] = {
            "status": "forgotten",
            "scope": removed.scope,
            "kind": removed.statement.kind,
            "text": removed.statement.text,
        }
        if removed.statement.recalls:
            forgotten["recalls_removed"] = removed.statement.recalls
        return forgotten
