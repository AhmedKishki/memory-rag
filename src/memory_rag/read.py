"""Answering one read: words, meaning, and only what matched.

This is the blocking path, so it is the one place where the cost rules are
strict. A read embeds its query once, scans the scope's vectors once, queries
the lexical index once, fuses the two sides, and returns. It does not walk the
scope's files, it does not embed anything but the query, and it does not load a
model: a lookup that has to wait for a load is a lookup that stalls its caller,
and the model is loaded on the serving side instead.

Three properties are decisions rather than details:

* **Both sides answer, and the answer says which.** A dense side alone is the
  weakest mode in the collection's own measurements, and its worst class is
  proper nouns, which is most of what a memory holds; a word side alone cannot
  find a statement that never used the caller's words. So a unit may arrive by
  words, by meaning, or by both, and it is labelled.
* **Nothing below the floor is admitted.** Without a floor a dense side always
  answers, always with the nearest thing it has, and always answering is worse
  than declining. A unit within the relative margin of the query's own best is
  the one exception, so a single strong match is not discarded with the rest.
* **New information wins a tie, and only a tie.** The document is ordered newest
  first, so a statement's position says when it was recorded, and a statement that
  matched equally well and was recorded later is the better answer. The
  preference is a small bonus on the fused score and nothing more: it breaks ties
  and can never lift a weaker match over a better one.
* **A read reports what it could not do.** If the model is not resident, the
  answer says the semantic side was unavailable rather than quietly returning a
  worse result that looks like the same one. If the index may be behind because
  a file was edited in place, it says that too, and names the command that
  fixes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .index import MemoryIndex
from .models import Embedder, ModelError, Reranker
from .vectors import VectorStore, cosine_similarity

if TYPE_CHECKING:
    # A type only: retrieval.py owns the object that calls in here, so importing
    # it at runtime would close a loop.
    from .retrieval import RetrievalSettings

__all__ = ["Answer", "answer_read"]


@dataclass(slots=True)
class Answer:
    """One read's result, and everything a caller is told about how it was got."""

    scope: str
    query: str
    units: list[dict[str, Any]] = field(default_factory=list)
    semantic_available: bool = False
    units_pending: int = 0
    truncated: bool = False
    superseded_removed: list[str] = field(default_factory=list)
    hint: str | None = None
    #: The record's own key, vector, and place for each unit, in the same order as
    #: `units`. A recall that merges two memories has to rank statements it did not
    #: fetch for this read and compare statements it read from another memory, so
    #: these travel with the answer rather than being looked up again; none of them
    #: is part of what the caller is told.
    unit_keys: list[str] = field(default_factory=list)
    unit_vectors: list[tuple[float, ...]] = field(default_factory=list)
    unit_scores: list[float] = field(default_factory=list)
    unit_stamps: list[int] = field(default_factory=list)
    #: What the read collapsed, and by which test, so a caller that is told three
    #: statements came back knows two of them said the same thing.
    collapsed: list[dict[str, Any]] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        """Return the object a read answers with.

        An answer carries a field when the field has news, and nothing else. The
        statements are the answer; every other field is either a fact the caller
        asked for by asking (``truncated`` says the list is a selection), a
        condition the caller has to act on (``units_pending``,
        ``semantic_available``), or a disclosure the caller could not otherwise
        account for (``superseded_removed``, ``collapsed_repetitions``). What a
        read leaves out is what it computed and the caller cannot use: the scores
        that ordered it, the side each statement was found by, how long it took,
        the position a statement holds in the document, and counts a caller can
        count for itself.
        """

        answer: dict[str, Any] = {"units": self.units}
        if self.truncated:
            answer["truncated"] = True
        if self.units_pending:
            answer["units_pending"] = self.units_pending
        if not self.semantic_available:
            answer["semantic_available"] = False
        if self.superseded_removed:
            answer["superseded_removed"] = self.superseded_removed
        if self.collapsed:
            answer["collapsed_repetitions"] = self.collapsed
        if self.hint is not None:
            answer["hint"] = self.hint
        return answer


def _normalized(text: str) -> str:
    """Return the words of a statement, so two wordings of it read the same."""

    return " ".join(str(text or "").split())


def collapse_repetitions(
    ranked: list[tuple[float, int, str, str, tuple[float, ...] | None, dict[str, Any]]],
    *,
    threshold: float,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the answer's statements with near-repeats of them removed.

    Two tests, in that order, and both over the answer's own candidates rather than
    over the memory. Words first, because a statement the record already holds
    word for word needs no model and no vector to recognise, and it is the case a
    caller is most likely to produce. Cosine second, for the statement that says
    the same thing in other words, which is the one words cannot see.

    A survivor is compared only against survivors, so the first — and therefore
    the best-ranked — statement of a repeated pair is the one kept, and the rest
    are dropped rather than shown beside it. The work is the number of statements
    the answer holds, which is the caller's `limit` and not the size of the memory,
    so a large memory costs no more to read than a small one.

    The dropped statements are returned as well, because a read that quietly
    returned three of five matches is a read the caller cannot account for.
    """

    kept: list[dict[str, Any]] = []
    kept_text: list[str] = []
    kept_vectors: list[tuple[float, ...]] = []
    collapsed: list[dict[str, Any]] = []
    for _score, _stamp, _key, text, vector, stated in ranked:
        normalized = _normalized(text)
        if any(normalized == held for held in kept_text):
            collapsed.append({**stated, "collapsed_by": "same words"})
            continue
        if vector is not None:
            score = _nearest(vector, kept_vectors, threshold)
            if score is not None:
                collapsed.append(
                    {**stated, "collapsed_by": "same meaning", "similarity": score}
                )
                continue
        if len(kept) >= max(int(limit), 1):
            break
        kept.append(stated)
        kept_text.append(normalized)
        if vector is not None:
            kept_vectors.append(vector)
    return kept, collapsed


def _nearest(
    vector: tuple[float, ...],
    others: list[tuple[float, ...]],
    threshold: float,
) -> float | None:
    """Return how alike this vector is to the closest of the others, if close enough."""

    best: float | None = None
    length = len(vector)
    for other in others:
        if len(other) != length:
            continue
        score = cosine_similarity(vector, other)
        if best is None or score > best:
            best = score
    return best if best is not None and best >= threshold else None


def merge_answers(
    answers: list[Answer],
    *,
    limit: int,
    duplicate_cosine: float = 0.99,
) -> Answer:
    """Combine one answer per scope into a single ranked answer.

    The scores are the same measure in every scope — a fused rank sum, with the
    recency bonus applied — so a statement is placed by how well it matched rather
    than by which memory it came from. A statement from the account's memory does
    not outrank a better match from the project's, and the other way round.

    The counts are reported per scope because a caller quoting a statement has to
    say which memory it came from, and that is a fact about the answer rather than
    something to work out afterwards.

    A repetition is collapsed here and not inside either scope's read, because the
    case that matters is one statement filed in both memories: it is one statement,
    and a caller told it twice is told the same thing about the same question. The
    best-ranked of a repeated pair is the one that survives, so the copy that
    matched best is the one the caller is shown.
    """

    combined = Answer(scope="both", query=answers[0].query if answers else "")
    ranked: list[
        tuple[float, int, str, str, tuple[float, ...] | None, dict[str, Any]]
    ] = []
    for answer in answers:
        vectors = dict(zip(answer.unit_keys, answer.unit_vectors, strict=False))
        for position, unit in enumerate(answer.units):
            # Every statement names the memory it came from: a caller quoting one
            # has to say which, and that is a fact about the answer. The ranking
            # travels beside the answer rather than inside it, in the lists parallel
            # to `units`, so the merged answer can order two memories together
            # without a caller ever seeing a score.
            stated = {"scope": answer.scope, **unit}
            key = answer.unit_keys[position] if position < len(answer.unit_keys) else ""
            score = (
                answer.unit_scores[position]
                if position < len(answer.unit_scores)
                else 0.0
            )
            stamp = (
                answer.unit_stamps[position]
                if position < len(answer.unit_stamps)
                else 0
            )
            ranked.append(
                (
                    float(score),
                    int(stamp),
                    key,
                    str(unit.get("text") or ""),
                    vectors.get(key),
                    stated,
                )
            )
    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    kept, collapsed = collapse_repetitions(
        ranked,
        threshold=float(duplicate_cosine),
        limit=max(int(limit), 1) + len(answers),
    )
    combined.units = kept[: max(int(limit), 1)]
    combined.collapsed = collapsed
    combined.truncated = any(answer.truncated for answer in answers) or (
        len(ranked) > max(int(limit), 1)
    )
    combined.units_pending = sum(answer.units_pending for answer in answers)
    combined.superseded_removed = [
        name for answer in answers for name in answer.superseded_removed
    ]
    combined.semantic_available = all(answer.semantic_available for answer in answers)
    if not combined.units:
        combined.hint = (
            "no statement matched those words, in this project or on the account. "
            "Try other words, or recall with a kind to see everything filed under "
            "one."
            if combined.semantic_available
            else "no statement matched those words, and the meaning side was not "
            "available: only the exact words would have found anything. Try other "
            "words."
        )
    return combined


def _fuse(
    lexical: Sequence[tuple[str, int]],
    dense: Sequence[tuple[str, float]],
    policy: RetrievalSettings,
) -> list[tuple[str, float, str, bool]]:
    """Return (unit_key, score, side, both) for the fused ranking.

    Weighted reciprocal rank fusion, as the collection's other server measured
    it: each side contributes ``weight / (rrf_k + rank)``, so the two are compared
    by position rather than by putting a BM25 score and a cosine on one scale.
    """

    scores: dict[str, float] = {}
    sides: dict[str, set[str]] = {}
    for rank, (key, _) in enumerate(lexical, start=1):
        scores[key] = scores.get(key, 0.0) + policy.bm25_weight / (policy.rrf_k + rank)
        sides.setdefault(key, set()).add("lexical")
    for rank, (key, _) in enumerate(dense, start=1):
        scores[key] = scores.get(key, 0.0) + policy.dense_weight / (policy.rrf_k + rank)
        sides.setdefault(key, set()).add("semantic")
    fused = [
        (key, score, "|".join(sorted(found)), len(found) == 2)
        for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for found in (sides[key],)
    ]
    return fused


def _recency_ranks(
    ordered: Sequence[tuple[str, float, str, bool]],
    history: dict[str, dict[str, object]],
) -> dict[str, int]:
    """Rank the dated statements among these candidates, newest first.

    The bonus is by date, not by position, because a document written before this
    package ordered statements on top holds its oldest statement first, and a
    position there means the opposite of what it means now. A statement with no
    recorded date — one typed into the Markdown by hand, or one from before the
    bookkeeping file existed — is not in this map at all, and so gets no bonus
    rather than a wrong one.
    """

    dated = [
        key
        for key, _score, _side, _both in ordered
        if (history.get(key) or {}).get("added_at")
    ]
    dated.sort(key=lambda key: str(history[key]["added_at"]), reverse=True)
    return {key: rank for rank, key in enumerate(dated)}


def _recency(rank: int | None, policy: RetrievalSettings) -> float:
    """Return how much a statement's date is worth against the ones it tied with.

    The newest statement among a set of otherwise equal matches is worth
    ``recency_bonus`` — ten per cent of its score by default — the next is worth
    half of that, and it decays from there. A statement that was never dated is
    worth nothing, which is the honest answer rather than a guess.

    This is a multiplier on a score that already says how well the statement
    matched. At the default it reorders matches that are close and leaves a
    materially better one first, which is what "a slight preference for what is
    new" has to mean if it is to mean anything.
    """

    if rank is None or policy.recency_bonus <= 0.0:
        return 0.0
    return policy.recency_bonus / (1 + rank)


def _stated(row: dict[str, str], facts: dict[str, Any]) -> dict[str, Any]:
    """Return one statement as the caller is told about it.

    Four facts an agent acts on: the words, the kind they were filed under, which
    memory holds them, and how often the memory has handed them back. The date is
    there only when the statement has one, and the position the document records it
    at is left out, because an answer is ordered and the order already says it.

    Every other fact the record keeps about a statement is bookkeeping this answer
    does not carry: the digests that identify it, the vectors that found it, the
    side that matched it, and the scores that put it where it is. A caller cannot
    act on any of them, and a statement's own words are what a caller reads.
    """

    stated: dict[str, Any] = {
        "text": row.get("text", ""),
        "kind": row.get("kind"),
        "recalls": int(facts.get("recalls") or 0),
    }
    added_at = facts.get("added_at")
    if added_at:
        stated["added_at"] = added_at
    return stated


def answer_read(
    *,
    scope: str,
    directory: Path,
    query: str,
    limit: int,
    embedder: Embedder,
    reranker: Reranker | None,
    policy: RetrievalSettings,
    kind: str | None = None,
) -> Answer:
    """Answer one read from a scope's own file.

    The index is open for the whole read, so the histories it holds are read from
    the same connection and the recalls this answer hands out are counted on it
    too. That is what keeps the work proportional to the answer: the counting is
    one ``UPDATE`` per unit returned, not a rewrite of anything.
    """

    answer = Answer(scope=scope, query=query)

    lexical: list[tuple[str, int]] = []
    semantic: list[tuple[str, float]] = []
    units_by_key: dict[str, dict[str, str]] = {}

    pool = max(int(limit) + 1, policy.pool_depth)
    identity = embedder.identity
    with MemoryIndex(directory) as index:
        # The document is an export of the record, so a read brings it back in
        # line and says whether it was stale: a hand edit to it is not read, and
        # the caller is told rather than left to find that their edit did not take.
        # A record with nothing in it adopts the document instead, which is how a
        # memory written by an older version is recovered.
        index.adopt_document_if_empty()
        answer.superseded_removed = index.retire_superseded()

        lexical = _lexical(index, query, pool, kind)
        units_by_key.update(index.units_by_key([key for key, _ in lexical]))

        with VectorStore(directory, identity.name, identity.dimension) as store:
            # Two counts, not a walk: the disclosure must not cost per unit.
            answer.units_pending = max(0, index.count_units() - store.count())
            if store.model_is_foreign():
                # Vectors from another model are not an answer, they are a
                # different question; the read says so and uses words.
                answer.semantic_available = False
            elif not getattr(embedder, "loaded", False):
                # Loading a model here would stall a caller mid-lookup. The read
                # says so instead, and the write side loads it for next time.
                answer.semantic_available = False
            else:
                try:
                    query_vector = embedder.embed_query(query)
                except ModelError:
                    answer.semantic_available = False
                else:
                    answer.semantic_available = True
                    semantic = store.search(
                        query_vector,
                        pool=policy.pool_depth,
                        floor=policy.cosine_floor,
                        margin=policy.relative_margin,
                    )
                    if kind is not None:
                        # The vector side is a set of numbers and has no column to
                        # filter, so a category is applied to what it found.
                        with MemoryIndex(directory) as types:
                            wanted = types.statements_by_kind(kind)
                        semantic = [
                            (key, score) for key, score in semantic if key in wanted
                        ]
                    units_by_key.update(
                        index.units_by_key([key for key, _ in semantic])
                    )

        fused = _fuse(lexical, semantic, policy)
        # Each side is asked for one row past the pool, so a longer fused list than
        # the caller asked for means the answer is a selection, not the whole match.
        answer.truncated = len(fused) > max(int(limit), 1)
        ordered = fused[: max(int(limit), 1)]

        if reranker is not None and len(ordered) > 1:
            # The cross-encoder reads `rerank_depth` of the candidates and no more,
            # which bounds what a read costs without touching the ranking: the
            # candidates it does not read keep their fused order below the ones it
            # does. Candidates and scores are built in one pass, so a score is never
            # paired with a statement it was not computed for.
            depth = max(1, int(policy.rerank_depth))
            rankable = [item for item in ordered if item[0] in units_by_key]
            candidates = [
                (item, units_by_key[item[0]]["text"]) for item in rankable[:depth]
            ]
            if candidates:
                texts = [text for _item, text in candidates]
                scores = reranker.score(query, texts)
                ranked_pairs = [
                    (item, score)
                    for (item, _text), score in zip(candidates, scores, strict=False)
                ]
                ranked_pairs.sort(key=lambda entry: entry[1], reverse=True)
                ordered = [item for item, _score in ranked_pairs] + rankable[depth:]

        # What was recalled has been recalled, whichever side found it. The count
        # is one update per unit answered, on the connection this read already
        # holds, so it is proportional to the answer rather than to the memory —
        # and SQLite makes it atomic, so nothing is written beside the memory and
        # nothing is deferred to a thread.
        keys = [key for key, _score, _side, _both in ordered if key in units_by_key]
        index.note_recalled(keys)
        history = index.history(keys)

        # Recency is applied last, over the fused and reranked order, so it settles
        # a tie that neither the two sides nor the cross-encoder could and never
        # overrides one of them. It is by recorded date rather than by position: a
        # document written before this package ordered statements on top holds its
        # oldest statement first, so a position there would mean the opposite of a
        # recency.
        ranks = _recency_ranks(ordered, history)
        if ranks:
            boosted = [
                (key, score * (1.0 + _recency(ranks.get(key), policy)), side, both)
                for key, score, side, both in ordered
            ]
            boosted.sort(key=lambda entry: entry[1], reverse=True)
            ordered = boosted

        answer.units = [
            {
                "scope": answer.scope,
                **_stated(units_by_key[key], history.get(key, {})),
            }
            for key, _score, _side, _both in ordered
            if key in units_by_key
        ]
        answer.unit_keys = [
            key for key, _score, _side, _both in ordered if key in units_by_key
        ]
        answer.unit_scores = [
            score for key, score, _side, _both in ordered if key in units_by_key
        ]
        answer.unit_stamps = [
            int(units_by_key[key].get("stamp") or 0)
            for key, _score, _side, _both in ordered
            if key in units_by_key
        ]
        # The vectors of the statements this read is holding, so a recall that
        # merges two memories can compare a statement from one against a statement
        # from the other without a second read.
        if answer.unit_keys:
            with VectorStore(directory, identity.name, identity.dimension) as store:
                held = store.vectors_for(answer.unit_keys)
            answer.unit_vectors = [held.get(key, ()) for key in answer.unit_keys]
    if not answer.units:
        # An answer with nothing in it is where an agent decides whether to try
        # again, so it says what to try rather than leaving the guidance in a
        # document the caller read once and is not reading now.
        answer.hint = (
            "no statement matched those words. Try other words for the same thing, "
            "or read with a type to see everything filed under one."
            if answer.semantic_available
            else "no statement matched those words, and only the words were "
            "searched: the meaning side was not available. Try other words."
        )
    return answer


def _lexical(
    index: MemoryIndex,
    query: str,
    pool: int,
    kind: str | None = None,
) -> list[tuple[str, int]]:
    """Return the statements the words found, best first, as (unit key, rank)."""

    found, _mode = index.search(query, pool, kind)
    return [(str(row["unit_key"]), rank) for rank, row in enumerate(found, start=1)]
