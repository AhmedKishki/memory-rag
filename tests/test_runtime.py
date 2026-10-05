"""Which projects a process serves, and in what order.

The first project is the active one, so it decides where a statement lands. The rest of
the record follows it, and a project that is already first must not appear a second
time: one memory listed twice is one memory searched twice, reported twice, and offered
to a caller as two scopes of the same name.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeEmbedder

from memory_rag.config import AccountConfig, global_directory
from memory_rag.registry import RegistryError, register
from memory_rag.retrieval import Retrieval, RetrievalSettings
from memory_rag.runtime import select_projects
from memory_rag.service import MemoryService


def _account(storage_root: Path) -> AccountConfig:
    return AccountConfig(
        storage_root=storage_root, global_directory=global_directory(storage_root)
    )


@pytest.fixture
def three_projects(tmp_path: Path, storage_root: Path) -> None:
    for name in ("alpha", "beta", "gamma"):
        (tmp_path / name / ".memory-rag").mkdir(parents=True)
        register(f"id-{name}", name, tmp_path / name)


def test_the_record_is_served_as_it_is(three_projects, storage_root: Path) -> None:
    account = _account(storage_root)
    assert [p.project_name for p in select_projects(account)] == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_the_named_project_is_first_and_appears_once(
    three_projects, storage_root: Path
) -> None:
    account = _account(storage_root)
    projects = select_projects(account, project_name="beta")
    assert [p.project_name for p in projects] == ["beta", "alpha", "gamma"]


def test_the_named_project_is_first_when_a_path_names_it(
    three_projects, storage_root: Path, tmp_path: Path
) -> None:
    account = _account(storage_root)
    projects = select_projects(account, project_root=str(tmp_path / "gamma"))
    assert [p.project_name for p in projects] == ["gamma", "alpha", "beta"]


def test_a_path_no_project_carries_is_refused(
    three_projects, storage_root: Path, tmp_path: Path
) -> None:
    account = _account(storage_root)
    with pytest.raises(RegistryError):
        select_projects(account, project_root=str(tmp_path / "nowhere"))


def test_a_name_no_project_carries_is_refused(
    three_projects, storage_root: Path
) -> None:
    account = _account(storage_root)
    with pytest.raises(RegistryError):
        select_projects(account, project_name="nowhere")


async def test_a_named_project_is_not_offered_as_two_scopes(
    three_projects, storage_root: Path
) -> None:
    """One project, one scope name, one memory searched once.

    A duplicate is not only a repeated row in a status answer: a recall would count the
    same statements twice, and the two scopes would share a name while holding the same
    directory.
    """

    account = _account(storage_root)
    service = MemoryService(
        account,
        Retrieval(
            embedder=FakeEmbedder(), policy=RetrievalSettings.from_settings(None)
        ),
        projects=select_projects(account, project_name="beta"),
    )
    try:
        names = [ref.name for ref in service.scopes()]
        assert names == ["global", "beta", "alpha", "gamma"]
        assert len(set(names)) == len(names)
        assert service.describe()["project_count"] == 3
        assert service.scope("beta").project == service.active_project()
        assert service.active_project_name() == "beta"
        assert sorted(service.recall_directories()) == [
            "alpha",
            "beta",
            "gamma",
            "global",
        ]
    finally:
        service.close()
