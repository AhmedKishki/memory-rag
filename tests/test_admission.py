"""What a statement may not say, and what the record does with what is left.

A memory that recorded whatever it was told records a row per session saying the
work was done, because a statement is known by a digest of its words and the date
is the part that differs between two accounts of one event. These are the rules
that stop it, and each refusal is stated here and checked, because a refusal that
is not tested is a refusal a later change can quietly widen.

The rows quoted here are real ones, taken from a live memory of 23 statements of
which 20 carried a date in their own words and 17 were reports of work approved.
Every rule below was measured against them rather than invented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memory_rag.read import merge_answers
from memory_rag.retrieval import RetrievalSettings
from memory_rag.service import MemoryService
from memory_rag.store import admission_failure, unit_key

#: One of the twenty statements in a live memory that carried a date in its words.
NARRATED = (
    "Section 4 T01: the author approved the exact critical-minimum revision "
    "on 2026-09-29."
)

#: And one of the two that carried none, and is still a statement worth holding.
HOLDS = (
    "Georgescu-Roegen, *The Entropy Law and the Economic Process* (HUP 1971) is "
    "in the corpus as `CS`."
)


# -- the rule on its own ----------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        NARRATED,
        "The author approved MAT-125 on 2026-09-29.",
        "T07 bridge ruling (author, 2026-09-29, updated by approved AK16 restoration).",
        "The build finished at 14:35 and the suite passed.",
    ],
)
def test_a_statement_carrying_a_date_is_refused(statement: str) -> None:
    """A date in a statement's own words is a moment, and the record dates every row."""

    failure = admission_failure(statement)
    assert failure is not None, f"accepted a statement reporting a moment: {statement}"


@pytest.mark.parametrize(
    "statement",
    [
        HOLDS,
        "In Corvellec (ed), *Waste as a Critique* (OUP 2025), each essay is mapped "
        "only when it is needed.",
        "The corpus holds 101 files, of which 91 are indexed.",
        "Keep the submodule pointer at the revision that shipped.",
        "Section 4 keeps Marx's categories as supporting vocabulary under Gidwani.",
        "The corpus is on generation 20260930T131800Z-5e4be986.",
    ],
)
def test_a_statement_saying_what_holds_is_accepted(statement: str) -> None:
    """A year on its own is not a date: which edition is meant still holds next year."""

    assert admission_failure(statement) is None


def test_the_refusal_names_where_the_statement_belongs() -> None:
    """A refusal a caller cannot act on is a refusal that stalls the caller."""

    failure = admission_failure(NARRATED) or ""
    assert "record_handoff" in failure
    assert "dates every statement itself" in failure


# -- the rule on the record path -------------------------------------------


async def test_recording_a_dated_statement_is_refused(service: MemoryService) -> None:
    """The date is refused by the tool, not only by the wording of the instructions."""

    from memory_rag.models import ModelError

    with pytest.raises(ModelError) as raised:
        await service.record(NARRATED, kind="DECISION", scope="local")
    assert "record_handoff" in str(raised.value)

    answer = await service.recall("critical-minimum revision", limit=10)
    assert answer["units"] == []


async def test_recording_what_holds_still_works(service: MemoryService) -> None:
    """The rule must not cost the memory the statements it is meant to keep."""

    await service.record(HOLDS, kind="RULE", scope="local")
    answer = await service.recall("Georgescu-Roegen", limit=10)
    assert [unit["text"] for unit in answer["units"]] == [HOLDS]


# -- the reserved kind ------------------------------------------------------


async def test_the_handoff_kind_is_reserved_for_the_handoff_tool(
    service: MemoryService,
) -> None:
    """A handoff replaces the previous one, so an ordinary record under it would vanish."""

    from memory_rag.models import ModelError

    with pytest.raises(ModelError) as raised:
        await service.record("A rule.", kind="HANDOFF", scope="local")
    assert "record_handoff" in str(raised.value)


async def test_a_handoff_still_replaces_the_one_before_it(service: MemoryService) -> None:
    """The reservation must not stop the one path that is meant to write that kind."""

    first = await service.handoff("Done and pushed.")
    second = await service.handoff("Done, and reindexed.")

    assert second["replaced"] == 1
    assert first["text"] == "Done and pushed."
    answer = await service.recall("reindexed", limit=10)
    assert [unit["text"] for unit in answer["units"]] == ["Done, and reindexed."]


# -- the vectors a rewording leaves behind ----------------------------------


def _store(service: MemoryService):
    """The vector store for this service's project memory."""

    from memory_rag.vectors import VectorStore

    identity = service.retrieval.embedder.identity
    return VectorStore(
        service.scope("local").directory, identity.name, identity.dimension
    )


async def test_a_write_drops_the_vectors_of_statements_it_no_longer_holds(
    service: MemoryService,
) -> None:
    """A live memory held 35 vectors for 23 statements, and nothing dropped the 12."""

    orphan = unit_key("RULE: the route table is read once, then cached")
    with _store(service) as store:
        store.put(orphan, (0.0, 0.0, 1.0), stamp="0", kind="RULE")
        assert store.has([orphan]) == {orphan}

    recorded = await service.record(
        "The route table is read once per process.", kind="RULE", scope="local"
    )
    assert recorded["vectors_removed"] == 1

    with _store(service) as store:
        assert store.has([orphan]) == set()
        assert store.count() == 0


async def test_a_settle_says_how_many_vectors_it_dropped(service: MemoryService) -> None:
    """An operator repairing a memory has to be able to account for what it removed."""

    orphan = unit_key("RULE: something the record no longer holds")
    with _store(service) as store:
        store.put(orphan, (0.0, 0.0, 1.0), stamp="0", kind="RULE")

    settled = await service.maintain("local")
    assert settled["vectors_removed"] == 1
    with _store(service) as store:
        assert store.has([orphan]) == set()


# -- the threshold near-duplicates are judged by ----------------------------


def test_the_duplicate_threshold_is_the_measured_one() -> None:
    """0.99 sat above every pair in the memory it was measured on, so it caught none."""

    assert RetrievalSettings(embedding_model="fake/model").duplicate_cosine == 0.85
    assert merge_answers.__kwdefaults__["duplicate_cosine"] == 0.85


def test_the_packaged_default_agrees_with_the_code() -> None:
    """A setting whose two homes disagree is settled by whichever a reader opened."""

    from memory_rag import settings as settings_module

    packaged = (
        Path(settings_module.__file__).parent / "default.toml"
    ).read_text(encoding="utf-8")
    assert "duplicate_cosine = 0.85" in packaged