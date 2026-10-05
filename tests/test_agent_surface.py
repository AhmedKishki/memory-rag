"""The agent surface's scope contract, read from the server rather than from the source.

The tools take no project argument, so a scope word is the only way a caller says where a
statement goes. `local` is the project this session is working in and `global` is the
account's, and those two words are the whole of what a scope-taking tool may be given. A
recorded project name is not accepted there: a tool that showed a project list would be a
tool the caller has to learn, and the session's repository is already known to the process
serving it.

These tests call the tools through the server, because a tool declared in a loop is
invisible to a read of the source and a change to the loop is what has to be caught.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes import FakeEmbedder

from memory_rag.config import AccountConfig, global_directory
from memory_rag.index import MemoryIndex
from memory_rag.registry import RegisteredProject
from memory_rag.retrieval import Retrieval, RetrievalSettings
from memory_rag.service import MemoryService
from memory_rag.store import KINDS
from memory_rag.surfaces.mcp import create_mcp


def _project(root: Path, name: str) -> RegisteredProject:
    directory = root / name
    (directory / ".memory-rag").mkdir(parents=True, exist_ok=True)
    return RegisteredProject(
        project_id=f"id-{name}",
        project_name=name,
        project_root=directory,
        registered_at="",
    )


@pytest.fixture
async def two_projects(tmp_path: Path, storage_root: Path):
    """A service whose active project is the first of two registered ones."""

    projects = [_project(tmp_path, "alpha"), _project(tmp_path, "beta")]
    service = MemoryService(
        AccountConfig(
            storage_root=storage_root,
            global_directory=global_directory(storage_root),
        ),
        Retrieval(
            embedder=FakeEmbedder(),
            policy=RetrievalSettings.from_settings(None),
        ),
        projects=projects,
    )
    try:
        yield service, projects
    finally:
        service.close()


def _statements(directory: Path) -> list[str]:
    with MemoryIndex(directory) as index:
        return [statement.text for statement in index.statements()]


async def test_the_local_word_files_in_the_project_this_session_is_in(
    two_projects,
) -> None:
    service, projects = two_projects
    server = create_mcp(service)
    await server.call_tool(
        "record_memory_rule", {"content": "a rule about this repo", "scope": "local"}
    )

    assert _statements(projects[0].project_root / ".memory-rag") == [
        "a rule about this repo"
    ]
    assert _statements(projects[1].project_root / ".memory-rag") == []


async def test_the_local_word_is_the_default_of_a_scope_taking_tool(
    two_projects,
) -> None:
    service, projects = two_projects
    server = create_mcp(service)
    await server.call_tool(
        "record_memory_rule", {"content": "a rule with no scope given"}
    )

    assert _statements(projects[0].project_root / ".memory-rag") == [
        "a rule with no scope given"
    ]


async def test_the_global_word_files_in_the_accounts_memory(two_projects) -> None:
    service, _projects = two_projects
    server = create_mcp(service)
    await server.call_tool(
        "record_memory_rule", {"content": "a rule about this repo", "scope": "global"}
    )

    assert _statements(service.account.global_directory) == ["a rule about this repo"]


async def test_a_handoff_takes_no_scope_and_goes_to_the_active_project(
    two_projects,
) -> None:
    """The tool offers no memory argument, so the word cannot redirect it."""

    service, projects = two_projects
    server = create_mcp(service)
    tool = next(
        tool
        for tool in await server.list_tools()
        if tool.name == "record_memory_handoff"
    )
    assert "scope" not in (tool.parameters.get("properties") or {})

    await server.call_tool(
        "record_memory_handoff", {"content": "where the work stands"}
    )

    assert _statements(projects[0].project_root / ".memory-rag") == [
        "where the work stands"
    ]
    assert _statements(projects[1].project_root / ".memory-rag") == []


@pytest.mark.parametrize("kind", [kind for kind in KINDS if kind.scope != "either"])
async def test_a_kind_with_a_fixed_memory_takes_no_scope_argument(
    two_projects, kind
) -> None:
    service, _projects = two_projects
    server = create_mcp(service)
    tool = next(
        tool
        for tool in await server.list_tools()
        if tool.name == f"record_memory_{kind.tool}"
    )
    assert "scope" not in (tool.parameters.get("properties") or {})


async def test_a_forget_naming_the_local_word_touches_one_memory(two_projects) -> None:
    """It does not widen to every memory, which is what an unmatched word used to do."""

    service, projects = two_projects
    server = create_mcp(service)
    for scope in ("local", "global"):
        await server.call_tool(
            "record_memory_rule",
            {"content": f"a statement filed in {scope}", "scope": scope},
        )
    await server.call_tool(
        "forget_memory", {"text": "a statement filed in local", "scope": "local"}
    )

    assert _statements(projects[0].project_root / ".memory-rag") == []
    assert _statements(service.account.global_directory) == [
        "a statement filed in global"
    ]


async def test_a_recall_carries_the_project_each_statement_came_from(
    two_projects,
) -> None:
    """A caller quoting a statement has to say which repository it is about."""

    service, _projects = two_projects
    server = create_mcp(service)
    await server.call_tool(
        "record_memory_rule",
        {"content": "the documentation is terse in this repository", "scope": "local"},
    )
    await server.call_tool(
        "record_memory_rule",
        {"content": "installations are done with the shared script", "scope": "global"},
    )

    local = await server.call_tool("recall_memory_rule", {"query": "terse repository"})
    shared = await server.call_tool("recall_memory_rule", {"query": "installations"})

    assert [unit["scope"] for unit in local.structured_content["units"]] == ["alpha"]
    assert [unit["scope"] for unit in shared.structured_content["units"]] == ["global"]


async def test_the_server_is_built_without_a_running_app(two_projects) -> None:
    """The surface is built before the app serves, so it takes no app state at all."""

    service, _projects = two_projects
    assert create_mcp(service) is not None
    await asyncio.sleep(0)
