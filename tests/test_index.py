"""The record: what a rebuild settles, and what it must leave alone.

`memory.sqlite3` is the record a memory is read from, so the only parts of it a rebuild
may touch are the ones derived from the statements. These tests state that boundary from
both sides: the statements survive, and the rendering a person opens is written again
rather than retired and forgotten.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest

from memory_rag.index import (
    INDEX_FILENAME,
    RENDERED_FILENAME,
    SUPERSEDED_FILENAMES,
    VECTOR_SCHEMA,
    MemoryIndex,
    index_path,
    read_unit_rows,
    rebuild,
    unit_key,
)
from memory_rag.store import parse_document

STATEMENTS = ("a statement worth keeping", "another one worth keeping")


def _populate(directory: Path) -> None:
    with MemoryIndex(directory) as index:
        for text in STATEMENTS:
            index.insert(text, "RULE")
        index.write_render()


def test_a_rebuild_keeps_every_statement(tmp_path: Path) -> None:
    directory = tmp_path / "scope"
    _populate(directory)

    report = rebuild(directory)

    assert (directory / INDEX_FILENAME).is_file()
    assert report["statements"] == len(STATEMENTS)
    with MemoryIndex(directory) as index:
        # The record orders statements oldest first, so the rendering they came from is
        # the reverse: what matters is that all of them are here.
        assert sorted(statement.text for statement in index.statements()) == sorted(
            STATEMENTS
        )


def test_a_rebuild_leaves_the_standing_document_in_place(tmp_path: Path) -> None:
    """Retiring a rendering removes the file, so the rendering is written again.

    The document is what a person opens to read a memory, and `rebuild` says it
    re-renders it. A build that removed the file and wrote nothing after it would have
    taken away the only readable form of the record and reported nothing.
    """

    directory = tmp_path / "scope"
    _populate(directory)
    rendered = directory / RENDERED_FILENAME

    rebuild(directory)

    assert rendered.is_file()
    text = rendered.read_text(encoding="utf-8")
    for statement in STATEMENTS:
        assert statement in text


def test_a_rebuild_writes_a_rendering_there_was_none_of(tmp_path: Path) -> None:
    directory = tmp_path / "scope"
    _populate(directory)
    (directory / RENDERED_FILENAME).unlink()

    rebuild(directory)

    assert (directory / RENDERED_FILENAME).is_file()


def test_a_rebuild_reports_where_it_wrote_the_rendering(tmp_path: Path) -> None:
    directory = tmp_path / "scope"
    _populate(directory)

    report = rebuild(directory)

    assert report["rendered"] == str(directory / RENDERED_FILENAME)


def test_a_statement_added_to_a_rendering_is_not_adopted(tmp_path: Path) -> None:
    """A rendering is written from the record, so it is not a second source of statements.

    A statement in the file that the record does not hold is one that was deliberately
    removed. Adopting it would undo the removal, and a forget or a console delete that
    came back on the next read is a memory that cannot be corrected.
    """

    directory = tmp_path / "scope"
    _populate(directory)
    with (directory / RENDERED_FILENAME).open("a", encoding="utf-8") as stream:
        stream.write("\n\nonly the rendering holds this\n")

    rebuild(directory)

    with MemoryIndex(directory) as index:
        texts = [statement.text for statement in index.statements()]
    assert "only the rendering holds this" not in texts
    assert (directory / RENDERED_FILENAME).is_file()
    assert "only the rendering holds this" not in (
        directory / RENDERED_FILENAME
    ).read_text(encoding="utf-8")


def test_a_record_that_never_held_a_statement_adopts_a_rendering(
    tmp_path: Path,
) -> None:
    """A memory written by an earlier version keeps its statements in the rendering.

    There is no record file beside it, so the rendering is the only copy there is.
    """

    directory = tmp_path / "scope"
    directory.mkdir(parents=True)
    (directory / RENDERED_FILENAME).write_text(
        "# MEMORY\n\nonly the rendering holds these\n\nand this one too\n",
        encoding="utf-8",
    )

    rebuild(directory)

    with MemoryIndex(directory) as index:
        texts = sorted(statement.text for statement in index.statements())
    assert texts == ["and this one too", "only the rendering holds these"]


def test_a_record_emptied_of_its_statements_does_not_adopt_a_rendering(
    tmp_path: Path,
) -> None:
    """Emptiness is not the condition; never having held a statement is.

    The rendering is the last thing the write that removed them did not rewrite, so
    reading it back would return the statement the caller just removed on the next
    read — and report it again, which is what a removal has to be able to do.
    """

    directory = tmp_path / "scope"
    _populate(directory)
    with sqlite3.connect(index_path(directory)) as connection:
        connection.execute("DELETE FROM unit")

    rebuild(directory)

    with MemoryIndex(directory) as index:
        assert index.statements() == []
    assert (directory / RENDERED_FILENAME).is_file()


def test_closing_an_index_keeps_what_it_wrote(tmp_path: Path) -> None:
    """The object's lifetime owns its transaction.

    SQLite holds a transaction open until something commits it, and closing a connection
    discards one that is still open. A write path that ends its block without committing
    would therefore lose its work silently, which is how a statement rescued from a
    rendering used to disappear while the rendering it came from was deleted.
    """

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("written inside the block", "RULE")
        assert len(index.statements()) == 1

    with MemoryIndex(directory) as index:
        assert [statement.text for statement in index.statements()] == [
            "written inside the block"
        ]


def test_closing_an_index_that_read_nothing_commits_nothing(tmp_path: Path) -> None:
    """The commit on close settles an open transaction; it does not open one."""

    directory = tmp_path / "scope"
    _populate(directory)
    before = (directory / INDEX_FILENAME).read_bytes()
    with MemoryIndex(directory) as index:
        index.statements()
        index.count_units()
    assert (directory / INDEX_FILENAME).read_bytes() == before


# -- what a file of an earlier shape is read as ----------------------------------


def _write_older_file(directory: Path, name: str, schema: str, rows: str) -> Path:
    """Write a file of an earlier shape, with its own statements in it."""

    path = directory / name
    connection = sqlite3.connect(path)
    try:
        connection.executescript(schema)
        connection.executescript(rows)
        connection.commit()
    finally:
        connection.close()
    return path


def test_a_statement_imported_from_a_retired_file_keeps_its_identity(
    tmp_path: Path,
) -> None:
    """The identity a file recorded is kept, because identity is kind and words.

    Recomputing it from the words alone drops the kind from the digest, so the imported
    row gets an identity no write will produce again and recording those same words under
    the same kind adds a second row: an answer gives the statement twice and removing it
    is refused as ambiguous.
    """

    directory = tmp_path / "scope"
    directory.mkdir(parents=True)
    text, kind = "the tests are run with one command", "RULE"
    _write_older_file(
        directory,
        SUPERSEDED_FILENAMES[0],
        "CREATE TABLE unit(text TEXT, unit_key TEXT, kind TEXT, stamp TEXT);",
        f"INSERT INTO unit VALUES ('{text}', '{unit_key(f'{kind}: {text}')}', '{kind}', '3');",
    )

    with MemoryIndex(directory) as index:
        index.retire_superseded()

    with MemoryIndex(directory) as index:
        assert [row.text for row in index.statements()] == [text]
        assert index.unit_keys() == {unit_key(f"{kind}: {text}")}
        index.insert(text, kind)
        assert index.count_units() == 1, "the same statement was filed twice"


def test_a_retired_file_of_vectors_is_not_read_as_statements(tmp_path: Path) -> None:
    """A vector row is a digest, not words, and a digest is not a statement.

    Reading one as a statement filed a 64-character key as the text of something a caller
    would then be shown and asked to reason about.
    """

    directory = tmp_path / "scope"
    directory.mkdir(parents=True)
    retired = _write_older_file(
        directory,
        SUPERSEDED_FILENAMES[1],
        VECTOR_SCHEMA,
        "INSERT INTO vector VALUES ('ba389deb26669f904b29b6bb847c4917c4"
        "1b632c3738000c1dcaf6ac80c0464a', '1', 'RULE', 'model', 3, x'00000000');",
    )
    assert read_unit_rows(retired) == []


def test_a_retired_file_lacking_an_identity_column_is_read_by_position(
    tmp_path: Path,
) -> None:
    """A missing column reads as blank rather than shifting the ones after it.

    Selecting only the columns a file happens to have and taking them by position put
    the statement's place where its kind belongs, and the statement was filed under a
    number.
    """

    directory = tmp_path / "scope"
    directory.mkdir(parents=True)
    retired = _write_older_file(
        directory,
        SUPERSEDED_FILENAMES[0],
        "CREATE TABLE unit(text TEXT, kind TEXT, stamp TEXT);",
        "INSERT INTO unit VALUES ('an older file without identities', 'RULE', '4');",
    )

    rows = read_unit_rows(retired)
    assert rows == [("an older file without identities", "", "RULE", "4", "", "")]

    with MemoryIndex(directory) as index:
        index.retire_superseded()
        statements = index.statements()
    assert [(row.text, row.kind) for row in statements] == [
        ("an older file without identities", "RULE")
    ]


def test_a_retired_file_without_a_kind_is_read_as_an_item(tmp_path: Path) -> None:
    """A blank kind becomes a real one rather than an empty column."""

    directory = tmp_path / "scope"
    directory.mkdir(parents=True)
    _write_older_file(
        directory,
        SUPERSEDED_FILENAMES[0],
        "CREATE TABLE unit(text TEXT, unit_key TEXT, stamp TEXT);",
        "INSERT INTO unit VALUES ('a statement with no kind', 'k1', '1');",
    )

    with MemoryIndex(directory) as index:
        index.retire_superseded()
        statements = index.statements()
    assert [(row.text, row.kind) for row in statements] == [
        ("a statement with no kind", "ITEM")
    ]


def test_a_file_that_cannot_be_read_is_not_a_file_that_holds_nothing(
    tmp_path: Path,
) -> None:
    """A caller deciding whether to delete a file has to be able to tell those apart."""

    missing = tmp_path / "absent.sqlite3"
    assert read_unit_rows(missing) is None
    empty = _write_older_file(
        tmp_path, "empty.sqlite3", "CREATE TABLE unit(text TEXT);", ""
    )
    assert read_unit_rows(empty) == []
    not_a_database = tmp_path / "rubbish.sqlite3"
    not_a_database.write_bytes(b"this is not a database" * 8)
    assert read_unit_rows(not_a_database) is None


# -- durability before removal ---------------------------------------------------


class _CommitWatched:
    """A connection whose commits are recorded, and can be made to fail.

    Everything but the commit is the real connection, reached through ``__getattr__``;
    the commit is this object's own method, so nothing forwards past it. Standing one
    in where a connection belongs is a lie the type checker cannot see and the index does
    not notice, because it only ever asks a connection for what a connection has.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        order: list[str] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self._connection = connection
        self._order = order if order is not None else []
        self._failure = failure

    def commit(self) -> None:
        self._order.append("commit")
        if self._failure is not None:
            raise self._failure
        self._connection.commit()

    def __getattr__(self, name: str):
        return getattr(self._connection, name)


def _watch_commits(
    index: MemoryIndex,
    order: list[str] | None = None,
    failure: Exception | None = None,
) -> _CommitWatched:
    """Give one index a connection that records its commits."""

    watched = _CommitWatched(index._open(), order, failure)
    index._connection = watched  # type: ignore[assignment]
    return watched


def test_a_commit_that_fails_leaves_the_file_that_held_the_statements(
    tmp_path: Path,
) -> None:
    """A file is removed only once the statements it held are committed.

    The record cannot say whether they are there, so SQLite has to. A commit that fails
    leaves the record without them and the file still holding them, and removing it there
    destroys the only copy; so the failure is raised with both files left as they were.
    """

    directory = tmp_path / "scope"
    directory.mkdir(parents=True)
    text = "a statement that exists in one file only"
    retired = _write_older_file(
        directory,
        SUPERSEDED_FILENAMES[0],
        "CREATE TABLE unit(text TEXT, unit_key TEXT, kind TEXT, stamp TEXT);",
        "INSERT INTO unit VALUES ('"
        + text
        + "', '"
        + unit_key(f"RULE: {text}")
        + "', 'RULE', '1');",
    )
    index = MemoryIndex(directory)
    _watch_commits(index, failure=sqlite3.OperationalError("disk I/O error"))
    try:
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            index.retire_superseded()
        assert retired.is_file(), "the only copy was removed before it was saved"
    finally:
        with contextlib.suppress(sqlite3.OperationalError):
            index.close()

    assert read_unit_rows(retired) == [
        (text, unit_key(f"RULE: {text}"), "RULE", "1", "", "")
    ]
    with MemoryIndex(directory) as reader:
        assert reader.statements() == [], "an unsaved adoption reached the record"


def test_a_commit_that_fails_on_close_is_raised_rather_than_swallowed(
    tmp_path: Path,
) -> None:
    """A write that cannot be saved is not a write that happened.

    Closing settles whatever the object still owes the file. A commit that fails leaves
    the transaction open and its statements uncertain; swallowing it would tell the
    caller the memory was written when the file does not hold it, and the next read is
    the only thing that would say otherwise. The unfinished transaction is rolled back
    rather than committed as a partial operation.
    """

    directory = tmp_path / "scope"
    index = MemoryIndex(directory)
    index.insert("a statement whose commit fails", "RULE")
    connection = index._open()
    connection.execute("UPDATE unit SET recalls = 7")
    assert connection.in_transaction, "the test needs work left unfinished"
    _watch_commits(index, failure=sqlite3.OperationalError("disk I/O error"))

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        index.close()
    assert index._connection is None, "the connection was not released"

    with MemoryIndex(directory) as reader:
        assert [row.recalls for row in reader.statements()] == [0]


def test_a_rolled_back_close_releases_the_connection(tmp_path: Path) -> None:
    """The object is closed either way, so a second close does nothing at all."""

    directory = tmp_path / "scope"
    index = MemoryIndex(directory)
    index.insert("a statement", "RULE")
    _watch_commits(index, failure=sqlite3.OperationalError("disk I/O error"))
    with pytest.raises(sqlite3.OperationalError):
        index.close()
    index.close()
    assert index._connection is None


def test_a_successful_close_settles_the_work_the_object_owes(tmp_path: Path) -> None:
    """The lifetime of the object owns its transaction."""

    directory = tmp_path / "scope"
    index = MemoryIndex(directory)
    index.insert("a statement", "RULE")
    index._open().execute("UPDATE unit SET recalls = 2")
    assert index._open().in_transaction

    index.close()

    with MemoryIndex(directory) as reader:
        assert [row.recalls for row in reader.statements()] == [2]


def _stage(index: MemoryIndex, text: str, key: str) -> None:
    """Stage one row without committing it, as a half-applied operation leaves it."""

    index._open().execute(
        "INSERT INTO unit(text, unit_key, kind, stamp, added_at, recalls)"
        " VALUES (?, ?, 'RULE', '0', '2026-01-01', 0)",
        (text, key),
    )
    assert index._open().in_transaction, "the test needs work left unfinished"


def test_a_failed_operation_commits_nothing_it_staged(tmp_path: Path) -> None:
    """A block that raised has not written, and must not leave half a write behind.

    `__exit__` closed the record the same way whether the block finished or raised, and
    closing commits. So an operation that failed part way through left the rows it had
    already staged sitting in the record as though they had been asked for: the caller
    was told it failed, the record said otherwise, and a later read answered with
    statements nobody asked to keep.
    """

    directory = tmp_path / "scope"
    index = MemoryIndex(directory)
    with pytest.raises(RuntimeError, match="the second half failed"), index:
        _stage(index, "the first half of a failed operation", "a-staged-key")
        raise RuntimeError("the second half failed")

    assert index._connection is None, "the connection was not released"
    with MemoryIndex(directory) as reader:
        assert reader.statements() == []


def test_a_failed_operation_leaves_the_statements_that_were_already_saved(
    tmp_path: Path,
) -> None:
    """The rollback undoes the block's own work, not the record's committed state."""

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("a statement that was saved", "RULE")

    with pytest.raises(ValueError), MemoryIndex(directory) as index:
        _stage(index, "a statement that was not", "another-staged-key")
        raise ValueError

    with MemoryIndex(directory) as reader:
        assert [row.text for row in reader.statements()] == [
            "a statement that was saved"
        ]


def test_a_failed_block_still_releases_the_connection(tmp_path: Path) -> None:
    """The connection is released on the way out of a failure as well as a success."""

    directory = tmp_path / "scope"
    index = MemoryIndex(directory)
    with pytest.raises(RuntimeError), index:
        raise RuntimeError
    assert index._connection is None
    index.close()


def test_a_write_is_settled_before_the_vectors_are_touched(tmp_path: Path) -> None:
    """A write is durable before derived: the record holds nothing unsettled.

    A statement the record no longer holds but a vector still describes is a state the
    two halves of the file disagree about, so the row's own transaction has to be closed
    before anything goes near the vectors.
    """

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("a statement about the ordering", "RULE")
        connection = index._open()
        assert not connection.in_transaction, "the row was not settled by the write"
        # A vector for a statement the record does not hold is what a reworded
        # statement leaves behind.
        connection.execute(
            "INSERT INTO vector(unit_key, stamp, kind, model, dimension, components)"
            " VALUES ('a-key-no-statement-has', '0', 'RULE', 'a-model', 1, x'00000000')"
        )
        connection.commit()
        assert index.collect_vectors() == 1
        assert not connection.in_transaction, "the dropped vector was left unsettled"


# -- what a person editing the document means -------------------------------------


def test_saving_the_document_unchanged_keeps_every_statement_as_it_was(
    tmp_path: Path,
) -> None:
    """A rendering carries no kinds, so a statement sent back is that statement.

    The page is handed `MEMORY.md`, whose lines have no kind on them, and the record
    holds kinds. Keying the history by identity alone meant the kind was dropped on the
    way back: the statement filed under `NOTE` came home as `ITEM`, with a new key, a
    new date, and a count of nothing — a save that changed nothing changed all of it,
    and the diff was empty because a rendering never carries a kind.
    """

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("a statement about the ledger", "NOTE")
        index.note_recalled([unit_key("NOTE: a statement about the ledger")])
        before = index.statements()

        answer = index.replace_all(parse_document(index.export()))

    assert answer == {
        "kept": 1,
        "added": 0,
        "removed": 0,
        "vectors_removed": 0,
    }
    with MemoryIndex(directory) as index:
        after = index.statements()
    assert [row.kind for row in after] == [row.kind for row in before]
    assert [row.key for row in after] == [row.key for row in before]
    assert [row.added_at for row in after] == [row.added_at for row in before]
    assert [row.recalls for row in after] == [row.recalls for row in before]


def test_a_kind_prefix_in_the_document_still_refiles_a_statement(
    tmp_path: Path,
) -> None:
    """Re-filing is deliberate when the file says which kind it means."""

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("a statement about the ledger", "NOTE")
        index.replace_all(
            parse_document("RULE: a statement about the ledger"),
        )
        assert [row.kind for row in index.statements()] == ["RULE"]


def test_a_document_holding_a_statement_twice_stores_it_once(tmp_path: Path) -> None:
    """The table has no constraint to ignore against, so the check is here."""

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        answer = index.replace_all(
            parse_document(
                "a statement about the ledger\n\na statement about the ledger"
            )
        )
    assert answer["kept"] == 0
    assert answer["added"] == 1
    with MemoryIndex(directory) as index:
        assert len(index.statements()) == 1


def test_a_statement_the_document_dropped_goes_with_its_history(tmp_path: Path) -> None:
    """The point of the edit is the removal, so the removal is complete."""

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("a statement kept", "NOTE")
        index.insert("a statement removed", "NOTE")
        answer = index.replace_all(parse_document("a statement kept"))

    assert answer["removed"] == 1
    with MemoryIndex(directory) as index:
        assert [row.text for row in index.statements()] == ["a statement kept"]
        assert index.history([unit_key("NOTE: a statement removed")]) == {}


def test_a_record_this_object_created_is_not_adopted_while_it_holds_statements(
    tmp_path: Path,
) -> None:
    """The condition is a record this process created *and* that holds nothing.

    The flag that says this object made the record file is not the whole condition, and
    the count is checked at the decision rather than left to the guard inside the import.
    This is the condition the other product on these files uses, and the two have to
    agree: they open the same record and a reader must not get a different answer from
    each of them.
    """

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("a statement about the ledger", "RULE")
        assert index.adopt_document_if_empty() == 0
        assert [row.text for row in index.statements()] == [
            "a statement about the ledger"
        ]


def test_a_record_created_by_this_object_still_adopts_a_rendering(
    tmp_path: Path,
) -> None:
    """The recovery path is unchanged: a file just created has held nothing."""

    directory = tmp_path / "scope"
    (directory).mkdir(parents=True)
    (directory / RENDERED_FILENAME).write_text(
        "# MEMORY\n\nonly the rendering holds these\n", encoding="utf-8"
    )
    with MemoryIndex(directory) as index:
        assert index.adopt_document_if_empty() == 1
        assert [row.text for row in index.statements()] == [
            "only the rendering holds these"
        ]
