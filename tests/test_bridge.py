"""The stdio bridge: what it does before it proxies anything.

The app holds every project's memory and the tools take no project argument, so the one
project a statement lands in is the one the app was started for. A client entry that
names another project would otherwise be answered out of a repository it did not ask for,
so the check happens here — before a pipe is proxied, and while the answer is still a
line a user can read.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from memory_rag import bridge
from memory_rag.config import AccountConfig, global_directory
from memory_rag.registry import RegisteredProject, register
from memory_rag.registry import load as load_projects


def _account(storage_root: Path) -> AccountConfig:
    return AccountConfig(
        storage_root=storage_root, global_directory=global_directory(storage_root)
    )


def test_a_name_this_machine_has_not_initialised_is_reported(
    storage_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """There is no memory to work in, and the message says how to make one.

    The refusal comes before any connection is attempted, so nothing is started and no
    session is registered for a client that could not be answered.
    """

    register("abc123", "research-rag", tmp_path / "repo")

    assert bridge.run("vanilla-ultra-rag-mcp", account=_account(storage_root)) == 1
    captured = capsys.readouterr()
    message = captured.err
    assert "vanilla-ultra-rag-mcp" in message
    assert "memory-rag init" in message
    assert captured.out == "", "the protocol stream carries diagnostics"


def test_a_refusal_quotes_the_project_name_for_a_shell(
    storage_root: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The remedy is a command a reader will paste, so the name is quoted as a shell would.

    An unquoted name is a command that means something else, or nothing, and the reader
    cannot tell which.
    """

    awkward = "a project's name"
    assert bridge.run(awkward, account=_account(storage_root)) == 1
    message = capsys.readouterr().err
    assert f"--name {bridge.shlex.quote(awkward)}" in message
    assert f"--project {bridge.shlex.quote(awkward)} start" in message


def test_the_project_the_app_is_serving_is_read_from_its_answer() -> None:
    """The app is asked, because only the app knows which project it treats as current."""

    control = SimpleNamespace(
        status=lambda: {
            "active_project": "research-rag",
            "active_project_root": "/srv/research-rag",
        }
    )
    assert bridge.serving_project(control) == {
        "name": "research-rag",
        "root": "/srv/research-rag",
    }


@pytest.mark.parametrize(
    "answer",
    [
        {"active_project": None},
        {},
        # A name with no root cannot be compared against a resolved project.
        {"active_project": "research-rag"},
        {"active_project": "research-rag", "active_project_root": ""},
    ],
)
def test_an_app_that_does_not_name_a_root_is_answered_as_serving_none(answer) -> None:
    assert (
        bridge.serving_project(SimpleNamespace(status=lambda: answer))["root"] is None
    )


def test_an_app_that_cannot_be_asked_is_answered_as_serving_none() -> None:
    def refuse() -> dict[str, object]:
        raise bridge.ControlError("connection closed")

    assert bridge.serving_project(SimpleNamespace(status=refuse)) == {
        "name": None,
        "root": None,
    }


def test_a_project_is_resolved_by_exact_name_only(
    storage_root: Path, tmp_path: Path
) -> None:
    """A substring of a recorded name is not a recorded name.

    A client entry carrying `research` must not be answered out of `research-rag`.
    """

    register("abc123", "research-rag", tmp_path / "repo")
    assert bridge.resolve_project_name("research", _account(storage_root)) is None
    found = bridge.resolve_project_name("research-rag", _account(storage_root))
    assert isinstance(found, RegisteredProject)
    assert found.project_name == "research-rag"


def test_a_recorded_name_is_what_a_client_entry_carries(
    storage_root: Path, tmp_path: Path
) -> None:
    """The name resolves to the project and never to a directory.

    An entry is written once and copied between machines, and a path in it is true on
    exactly one of them.
    """

    project = register("abc123", "research-rag", tmp_path / "repo")
    found = bridge.resolve_project_name("research-rag", _account(storage_root))
    assert found == project
    assert found.project_name in {item.project_name for item in load_projects()}
