"""The project record: what a command center knows, and how it resolves a name.

A client configuration is written once and copied between machines, so the thing a
configuration names has to be a project's name rather than its path. These tests cover
the resolution, and cover that a record is a pointer rather than a state cache: a
project can be moved or deleted while the pointer stays, and the pointer has to say so
rather than serve something.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_rag import registry
from memory_rag.registry import RegisteredProject, RegistryError


def _register(root: Path, name: str, project_id: str = "id-1") -> RegisteredProject:
    return registry.register(project_id, name, root)


def test_a_recorded_project_is_found_by_its_name_and_its_id(tmp_path: Path) -> None:
    _register(tmp_path / "repo", "research")
    assert [p.project_name for p in registry.load()] == ["research"]
    assert registry.resolve("research").project_name == "research"
    assert registry.resolve("id-1").project_name == "research"
    assert registry.resolve("RESEARCH").project_name == "research"


def test_the_name_matches_exactly_and_never_as_a_substring(tmp_path: Path) -> None:
    """A substring is a guess about a name, and the record holds the name.

    Resolving `res` to `research` would put a session's memory in a repository its
    caller did not ask for, which is worse than refusing.
    """

    _register(tmp_path / "repo", "research")
    assert registry.matches("res") == []
    with pytest.raises(RegistryError) as raised:
        registry.resolve("res")
    assert "No registered project" in str(raised.value)


def test_resolving_a_name_only_never_matches_another_project_id(tmp_path: Path) -> None:
    """A name and a different project's id may not stand for the same directory."""

    _register(tmp_path / "one", "alpha", project_id="beta")
    assert registry.named("beta") == []
    assert registry.resolve("beta").project_name == "alpha"


def test_registering_the_same_root_twice_updates_rather_than_duplicates(
    tmp_path: Path,
) -> None:
    _register(tmp_path / "repo", "old-name")
    _register(tmp_path / "repo", "new-name")
    assert [p.project_name for p in registry.load()] == ["new-name"]


def test_a_project_that_moved_is_followed_to_where_it_went(tmp_path: Path) -> None:
    """Re-registering a root replaces the record rather than adding a second one."""

    _register(tmp_path / "here", "demo")
    _register(tmp_path / "there", "demo")
    projects = registry.load()
    assert len(projects) == 1
    assert projects[0].project_root == (tmp_path / "there").resolve()


def test_two_projects_with_one_name_are_a_refusal_not_a_guess(tmp_path: Path) -> None:
    _register(tmp_path / "one", "demo", project_id="a")
    _register(tmp_path / "two", "demo", project_id="b")
    with pytest.raises(RegistryError) as raised:
        registry.resolve("demo")
    assert "more than one" in str(raised.value)
    # The id is how a caller with two projects of one name says which it meant.
    assert registry.resolve("a").project_id == "a"


def test_forgetting_a_project_removes_only_that_record(tmp_path: Path) -> None:
    _register(tmp_path / "one", "alpha", project_id="a")
    _register(tmp_path / "two", "beta", project_id="b")
    assert registry.forget("a") is True
    assert registry.forget("a") is False
    assert [p.project_name for p in registry.load()] == ["beta"]


def test_a_record_says_whether_the_project_it_points_at_is_there(
    tmp_path: Path,
) -> None:
    """A pointer can outlive what it points at, so a caller is told rather than
    trusting the number in a file."""

    project = tmp_path / "repo"
    _register(project, "demo")
    assert registry.load()[0].initialised() is False

    (project / ".memory-rag").mkdir(parents=True)
    (project / ".memory-rag" / "project.json").write_text("{}", encoding="utf-8")
    assert registry.load()[0].initialised() is True


def test_an_unreadable_record_does_not_hide_the_projects_that_are_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One entry with the wrong shape is skipped rather than guessed at, and rather
    than taking every project with it."""

    _register(tmp_path / "repo", "alpha")
    path = registry.registry_path()
    document = json.loads(path.read_text(encoding="utf-8"))
    document["projects"].append({"project_id": "broken"})
    document["projects"].append("not even an object")
    path.write_text(json.dumps(document), encoding="utf-8")
    assert [p.project_name for p in registry.load()] == ["alpha"]


def test_a_record_that_is_not_an_object_is_refused_rather_than_read(
    tmp_path: Path,
) -> None:
    path = registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RegistryError) as raised:
        registry.load()
    assert "not an object" in str(raised.value)


def test_the_record_lives_beside_the_settings_the_frozen_product_also_reads() -> None:
    """One account, one directory, for both products.

    Renaming it would strand every existing memory and split the project list between
    two files that could disagree about the same project.
    """

    from memory_rag.settings import USER_CONFIG_DIRECTORY

    assert registry.registry_path().parent.name == USER_CONFIG_DIRECTORY
    assert USER_CONFIG_DIRECTORY == "memory-ultra-rag-mcp"
    assert registry.REGISTRY_FILE == "projects.json"


def test_the_record_never_holds_a_statement(tmp_path: Path) -> None:
    """A pointer holds an address and an identity, and nothing a memory owns.

    A record that cached a scope's contents would be a second copy of a memory that
    the app would have to keep in step with the first.
    """

    _register(tmp_path / "repo", "demo")
    text = registry.registry_path().read_text(encoding="utf-8")
    for forbidden in ("memory.sqlite3", "statements", "recalls", "vector"):
        assert forbidden not in text


def test_a_written_record_is_never_left_half_written(tmp_path: Path) -> None:
    """A reader must not see a partial file, so the write is atomic."""

    _register(tmp_path / "repo", "demo")
    leftovers = list(registry.registry_path().parent.glob("*.tmp"))
    assert leftovers == []
    document = json.loads(registry.registry_path().read_text(encoding="utf-8"))
    assert document["schema_version"] == registry.SCHEMA_VERSION
