"""The service: which memory a name reaches, and what a caller is told about it.

Every surface reaches this object, so the rules below are the ones a browser, an agent,
and a terminal all share. A scope name that resolves to the wrong repository puts a
statement about one in another, which is a statement that is wrong everywhere else, and
a bound that lives in one surface rather than here is a bound the other two do not have.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from memory_rag.config import AccountConfig, global_directory
from memory_rag.models import ModelError
from memory_rag.registry import RegisteredProject, RegistryError, load, register
from memory_rag.retrieval import Retrieval, RetrievalSettings
from memory_rag.service import MAX_RECALL_LIMIT, MemoryService


def _project(root: Path, name: str) -> RegisteredProject:
    """Register one project on disk, with the directory its local memory lives in."""

    directory = root / name
    (directory / ".memory-rag").mkdir(parents=True, exist_ok=True)
    return RegisteredProject(
        project_id=f"id-{name}",
        project_name=name,
        project_root=directory,
        registered_at="",
    )


@pytest.fixture
def many_projects(tmp_path: Path) -> list[RegisteredProject]:
    return [_project(tmp_path, name) for name in ("alpha", "beta", "gamma")]


@pytest.fixture
def two_projects(tmp_path: Path) -> list[RegisteredProject]:
    """Projects none of which is called `local`, so the alias is free to mean the active."""

    return [_project(tmp_path, name) for name in ("alpha", "beta")]


def _service(storage_root: Path, projects: list[RegisteredProject]) -> MemoryService:
    return MemoryService(
        AccountConfig(
            storage_root=storage_root,
            global_directory=global_directory(storage_root),
        ),
        Retrieval(
            embedder=FakeEmbedder(), policy=RetrievalSettings.from_settings(None)
        ),
        projects=projects,
    )


@pytest.fixture
def several(many_projects, storage_root: Path):
    """A service over three projects, none of which is called `local` or `global`."""

    service = _service(storage_root, many_projects)
    yield service
    service.close()


@pytest.fixture
def plain(two_projects, storage_root: Path):
    """A service over two projects whose names do not collide with the aliases."""

    service = _service(storage_root, two_projects)
    yield service
    service.close()


@pytest.fixture
def unnamed(tmp_path: Path, storage_root: Path) -> list[RegisteredProject]:
    """Two projects recorded under one name, which a client's entry cannot address."""

    return [_project(tmp_path / "one", "twin"), _project(tmp_path / "two", "twin")]


# -- which memory a name reaches ------------------------------------------------


def test_a_project_may_not_be_recorded_under_either_product_word(
    tmp_path: Path,
) -> None:
    """`local` and `global` are how the product addresses a memory, not a project.

    A project recorded under one of them would answer to the word the agent tools and
    the command line pass for the session's own memory, and a tool that takes no project
    argument could not say which memory it wrote to.
    """

    for name in ("local", "LOCAL", "Global"):
        with pytest.raises(RegistryError) as raised:
            register("an-id", name, tmp_path / "repo")
        assert "local" in str(raised.value)
        assert "global" in str(raised.value)
    # Nothing was recorded, so no name was reserved either.
    assert load() == []


async def test_a_handoff_goes_to_the_active_project_whatever_else_is_registered(
    several: MemoryService,
) -> None:
    """A handoff is about this project's work, whatever the caller's scope word was.

    The tools take no scope argument for this kind at all, so the word cannot reach it:
    the project is resolved rather than the name spelled.
    """

    answer = await several.record("where the work stands", kind="HANDOFF")
    assert answer["scope"] == "alpha"


async def test_a_handoff_replaces_only_the_active_projects_handoff(
    several: MemoryService,
) -> None:
    await several.record("the first handoff", kind="HANDOFF")
    second = await several.handoff("the second handoff")
    assert second["scope"] == "alpha"
    active = await several.recall("handoff", kind="HANDOFF", limit=5)
    assert [unit["text"] for unit in active["units"]] == ["the second handoff"]


def test_the_active_project_is_still_reachable_by_both_spellings(
    plain: MemoryService,
) -> None:
    """With no project called `local`, the alias is the active project's memory."""

    assert plain.active_project_name() == "alpha"
    assert plain.scope("local").directory == plain.scope("alpha").directory
    assert plain.scope("local").project == plain.active_project()


def test_a_name_no_project_carries_is_refused_with_the_ones_that_exist(
    several: MemoryService,
) -> None:
    with pytest.raises(RegistryError) as raised:
        several.scope("delta")
    message = str(raised.value)
    for known in ("global", "local", "alpha", "beta", "gamma"):
        assert known in message


def test_one_name_carrying_two_projects_is_refused(
    storage_root: Path, unnamed: list[RegisteredProject]
) -> None:
    """A name that stands for two directories is a name a client entry cannot resolve.

    Choosing one of them silently would put a session's memory in a repository its
    caller did not name, so the refusal names both roots and the command that fixes it.
    """

    service = _service(storage_root, unnamed)
    try:
        with pytest.raises(RegistryError) as raised:
            service.scope("twin")
        message = str(raised.value)
        assert str(unnamed[0].project_root) in message
        assert str(unnamed[1].project_root) in message
    finally:
        service.close()


def test_the_status_answer_names_the_project_the_tools_file_into(
    several: MemoryService,
) -> None:
    """The tools take no project argument, so the answer has to say which one they use."""

    assert several.describe()["active_project"] == "alpha"


# -- the bound on an answer -----------------------------------------------------


@pytest.mark.parametrize("limit", [0, -1, MAX_RECALL_LIMIT + 1, 100_000])
async def test_a_recall_outside_the_bound_is_refused(
    plain: MemoryService, limit
) -> None:
    """The bound is the service's, and it refuses rather than clamping.

    A caller that asked for a thousand statements and received ten has an answer it
    cannot account for, and the control channel's own bound has to be this one.
    """

    with pytest.raises(ModelError) as raised:
        await plain.recall("anything", limit=limit)
    assert str(MAX_RECALL_LIMIT) in str(raised.value)


async def test_a_recall_inside_the_bound_is_answered(plain: MemoryService) -> None:
    answer = await plain.recall("anything", limit=1)
    assert "units" in answer


# -- what a write reports -------------------------------------------------------


async def test_recording_names_the_memory_it_filed_in(plain: MemoryService) -> None:
    """`local` is a way of addressing a project's memory rather than its name.

    The answer carries the name it actually reached, so a caller that asked for `local`
    can say which repository it got.
    """

    answer = await plain.record("a rule about this repository", kind="RULE")
    assert answer["scope"] == "alpha"
    assert (await plain.recall("repository", limit=1))["units"][0]["scope"] == "alpha"


async def test_recording_into_a_named_project_reports_that_name(plain) -> None:
    answer = await plain.record(
        "a rule about that repository", kind="RULE", scope="beta"
    )
    assert answer["scope"] == "beta"


async def test_a_kind_that_names_its_memory_overrides_the_scope(
    plain: MemoryService,
) -> None:
    answer = await plain.record(
        "how the user wants to be spoken to", kind="PREFERENCE", scope="local"
    )
    assert answer["scope"] == "global"


# -- a forget is not widened by a name that means nothing ------------------------


async def test_a_forget_naming_no_memory_is_refused(plain: MemoryService) -> None:
    """An unrecognised scope must not be read as "every memory".

    The retrieval searches whatever it is handed, so a name it cannot match would fall
    back to a search of every memory and could remove a statement out of a repository the
    caller named something else for.
    """

    await plain.record("a claim about the ledger", kind="RULE", scope="beta")
    with pytest.raises(RegistryError):
        await plain.forget("a claim about the ledger", scope="gamma")
    assert (await plain.recall("ledger", limit=1))["units"]


async def test_a_forget_naming_a_memory_searches_only_that_one(
    plain: MemoryService,
) -> None:
    await plain.record("a claim about the ledger", kind="RULE", scope="beta")
    answer = await plain.forget("a claim about the ledger", scope="beta")
    assert answer["scope"] == "beta"
    assert answer["status"] == "forgotten"


async def test_a_forget_naming_no_scope_searches_every_memory(
    plain: MemoryService,
) -> None:
    await plain.record("a claim about the ledger", kind="RULE", scope="beta")
    answer = await plain.forget("a claim about the ledger")
    assert answer["scope"] == "beta"


# -- a statement whose vector is not ready ----------------------------------------


class _NoModel:
    """An embedder whose model is not available, which is a state the product supports.

    The statements are recorded and their vectors are not, the read says so in
    ``units_pending``, and the worker keeps the statements queued for a later attempt.
    """

    def __init__(self) -> None:
        self._delegate = FakeEmbedder()

    @property
    def identity(self):
        return self._delegate.identity

    @property
    def loaded(self) -> bool:
        return False

    def warm(self) -> None:
        raise ModelError("no embedding model is available here")

    def embed_documents(self, texts):
        raise ModelError("no embedding model is available here")

    def embed_query(self, text):
        raise ModelError("no embedding model is available here")


async def _answer_without_vectors(
    duplicate_cosine: float, storage_root: Path, tmp_path: Path
) -> dict:
    """Two distinct statements in a memory holding no vectors, read at one threshold."""

    account = AccountConfig(
        storage_root=storage_root, global_directory=global_directory(storage_root)
    )
    service = MemoryService(
        account,
        Retrieval(
            embedder=_NoModel(),
            policy=replace(
                RetrievalSettings.from_settings(None), duplicate_cosine=duplicate_cosine
            ),
        ),
        projects=[_project(tmp_path, "alpha")],
    )
    try:
        await service.record("the build is run with one command", kind="RULE")
        await service.record("the deployment is run with one command", kind="RULE")
        return await service.recall("run command", limit=5)
    finally:
        service.close()


async def test_a_statement_with_no_vector_is_not_collapsed_as_a_repetition(
    storage_root: Path, tmp_path: Path
) -> None:
    """There is nothing to compare, so only the words test applies to it.

    A vector that is missing arrives as an empty tuple, and an empty tuple compared with
    another one has a similarity of nothing. With the cosine threshold at or below zero
    every statement without a vector folded into whichever such statement came before
    it, reported as a repetition of it, while the answer said two statements were still
    waiting to be embedded.
    """

    answer = await _answer_without_vectors(0.0, storage_root, tmp_path)
    assert answer["units_pending"] == 2
    assert len(answer["units"]) == 2
    assert answer.get("collapsed_repetitions", []) == []


@pytest.mark.parametrize(
    "limit",
    [
        True,
        False,
        1.5,
        0.5,
        "10",
        None,
        [10],
        complex(1, 0),
    ],
)
async def test_a_limit_that_is_not_a_whole_number_is_refused(
    plain: MemoryService, limit
) -> None:
    """`bool` is an `int` in Python and a float is not one, so converting alone is not a check.

    `int(True)` is one statement and `int(1.5)` is one, so a caller asking for half a
    statement would have been answered with an answer it cannot account for.
    """

    with pytest.raises(ModelError) as raised:
        await plain.recall("anything", limit=limit)
    assert str(MAX_RECALL_LIMIT) in str(raised.value)


@pytest.mark.parametrize("limit", [1, 2, 49, MAX_RECALL_LIMIT])
async def test_a_limit_inside_the_bound_is_answered(
    plain: MemoryService, limit
) -> None:
    answer = await plain.recall("anything", limit=limit)
    assert "units" in answer


@pytest.mark.parametrize("limit", [0, -1, MAX_RECALL_LIMIT + 1, 10**9])
async def test_a_limit_outside_the_bound_is_refused(
    plain: MemoryService, limit
) -> None:
    with pytest.raises(ModelError) as raised:
        await plain.recall("anything", limit=limit)
    assert str(MAX_RECALL_LIMIT) in str(raised.value)
