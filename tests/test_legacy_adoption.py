"""Legacy recovery must not undo an intentional record edit."""

import sqlite3
from pathlib import Path

import pytest

from memory_rag.index import MemoryIndex, index_path
from memory_rag.store import unit_key


def legacy_document(directory: Path) -> None:
    with MemoryIndex(directory.parent / "seed") as seed:
        seed.insert("Prefer spaces", "RULE")
        text = seed.export()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "MEMORY.md").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("clear", ["drop", "replace"])
def test_same_object_does_not_recover_a_statement_it_removed(
    tmp_path: Path, clear: str
) -> None:
    with MemoryIndex(tmp_path) as index:
        key, _ = index.insert("Prefer spaces", "RULE")
        index.write_render()
        if clear == "drop":
            index.drop(key)
        else:
            index.replace_all([])
        assert index.adopt_document_if_empty() == 0
        assert index.count_units() == 0


def test_a_recovered_record_cannot_be_recovered_again_after_forget(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "memory"
    legacy_document(directory)
    with MemoryIndex(directory) as index:
        assert index.adopt_document_if_empty() == 1
        index.drop(next(iter(index.unit_keys())))
        assert index.adopt_document_if_empty() == 0


@pytest.mark.parametrize("authoritative", [False, True])
def test_upgrade_distinguishes_a_cache_from_an_emptied_record(
    tmp_path: Path, authoritative: bool
) -> None:
    directory = tmp_path / "memory"
    legacy_document(directory)
    with sqlite3.connect(index_path(directory)) as connection:
        extra = ", added_at, recalls" if authoritative else ""
        connection.execute(
            "CREATE VIRTUAL TABLE unit USING fts5("
            f"text, unit_key UNINDEXED, kind UNINDEXED, stamp UNINDEXED{extra})"
        )
        connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "INSERT INTO meta VALUES ('schema_version', ?)",
            ("7" if authoritative else "5",),
        )
    with MemoryIndex(directory) as index:
        assert index.adopt_document_if_empty() == (0 if authoritative else 1)


def test_upgrade_fills_missing_columns_without_shifting_kind_and_stamp(
    tmp_path: Path,
) -> None:
    with sqlite3.connect(index_path(tmp_path)) as connection:
        connection.execute("CREATE VIRTUAL TABLE unit USING fts5(text, kind, stamp)")
        connection.execute("INSERT INTO unit VALUES ('Prefer spaces', 'RULE', '4')")
        connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO meta VALUES ('schema_version', '5')")
    with MemoryIndex(tmp_path) as index:
        statement = index.statements()[0]
        assert statement.kind == "RULE"
        assert statement.position == 4
        assert statement.key == unit_key("RULE: Prefer spaces")
        index.insert("Prefer spaces", "RULE")
        assert index.count_units() == 1


def test_direct_adoption_opens_the_record_before_checking_eligibility(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "memory"
    legacy_document(directory)
    index = MemoryIndex(directory)
    try:
        assert index.adopt_document_if_empty() == 1
    finally:
        index.close()
