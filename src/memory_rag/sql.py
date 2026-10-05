"""The SQL console: a read that is unrestricted and a write that is not.

``memory.sqlite3`` is not a derived cache. It holds the statements, the kind each is
filed under, its stamp, its date, its recall count, the word index, and the vectors,
and it is the record a memory is read from. The ``unit`` table is an FTS5 virtual
table, so a statement's text and the words that find it are the same rows: a hand edit
cannot put them out of step with each other. The one thing a hand edit *can* leave
behind is a vector for a statement that now says something else, which is why an
accepted write drops the vectors of the statements it touched and lets the worker
embed them again.

So the read path is free. A ``SELECT`` runs against a connection SQLite itself holds
read-only, and a user who wants to know what a memory holds can ask it anything. The
write path is bounded to statements for the reason above: a write that dropped a
table, changed the schema version, or edited the vector table directly would leave a
memory whose words and whose vectors describe different records, which is a memory
that answers wrongly rather than one that fails loudly.

A read SQLite cannot run and a write that would leave a row the record cannot count are
both refusals in this module's words: a raw ``sqlite3.Error`` reaches the terminal as a
traceback and the browser as a 500, naming neither the statement nor what to write
instead.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .index import MemoryIndex, index_path

#: The one table a write may touch. Everything else in the file is either derived from
#: a statement or is the record's own bookkeeping.
WRITABLE_TABLE = "unit"

#: The tables a write may not touch, named here so a refusal can say which one it was
#: rather than only that it was refused.
PROTECTED_TABLES = (
    "meta",
    "vector",
    "unit_data",
    "unit_idx",
    "unit_content",
    "unit_docsize",
    "unit_config",
)

#: A read answers with at most this many rows. A memory is read by a person in a
#: browser, and a query that returns a hundred thousand rows is not an answer.
MAXIMUM_ROWS = 500

#: A read or a write longer than this is refused rather than truncated, so a paste
#: that lost its end fails loudly instead of running a statement the user cannot see.
MAXIMUM_STATEMENT = 20_000

#: The leading keyword of a statement, and which of them may write.
_WRITES = ("insert", "update", "delete", "replace")
_READS = ("select", "with", "explain", "values")

#: The keywords a write is never allowed to contain, whatever it claims to target.
#: They are refused by name because each of them can reach past the statements.
_FORBIDDEN = (
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "create",
    "alter",
    "drop",
    "truncate",
)

_TARGET = re.compile(
    r"^\s*(?:insert\s+or\s+\w+\s+into|insert\s+into|replace\s+into|update|delete\s+from)\s+"
    r"(?:\"?([A-Za-z_][A-Za-z0-9_]*)\"?|main\s*\.\s*\"?([A-Za-z_][A-Za-z0-9_]*)\"?)",
    re.IGNORECASE,
)
_LEADING = re.compile(r"^\s*(?:--[^\n]*\n|/\*.*?\*/\s*)*([A-Za-z]+)", re.DOTALL)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SqlRefusal(ValueError):
    """Raised when a statement is one this console will not run.

    The message names the statement's leading keyword or the table it reached for,
    because a refusal a user cannot act on is a refusal they retry.
    """


def sql_query(scope: Any, statement: str) -> dict[str, Any]:
    """Run a read against one memory, and answer with the rows it returned.

    The connection is held read-only by SQLite rather than by this module inspecting
    what came back, so the guarantee does not depend on the classifier below agreeing
    with the parser: a read that reached a write would fail in the engine.
    """

    text = _checked(statement)
    _require_read(text)
    path = index_path(scope.directory)
    if not path.is_file():
        return {
            "scope": scope.name,
            "statement": text,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "empty_memory": True,
        }
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        try:
            cursor = connection.execute(text)
            columns = [description[0] for description in cursor.description or []]
            rows = [list(row) for row in cursor.fetchmany(MAXIMUM_ROWS + 1)]
        except sqlite3.Error as error:
            raise SqlRefusal(f"SQLite could not run that read: {error}") from error
    finally:
        connection.close()
    truncated = len(rows) > MAXIMUM_ROWS
    answer: dict[str, Any] = {
        "scope": scope.name,
        "statement": text,
        "columns": columns,
        "rows": rows[:MAXIMUM_ROWS],
        "row_count": min(len(rows), MAXIMUM_ROWS),
    }
    if truncated:
        answer["truncated"] = True
    return answer


def sql_execute(scope: Any, statement: str) -> dict[str, Any]:
    """Run a write against one memory's statements, and report what it touched.

    The statements a write touched are found by comparing the record before and after,
    not by reading its own ``WHERE`` clause: a ``WHERE`` that was wrong once will be
    wrong again, and guessing wrong leaves a vector describing a statement that is gone.

    A write is refused before it is committed if it leaves a row the record cannot read
    back, because the record counts a statement's place and its recall count and every
    later read of a row like that fails. The whole record is checked, which is one more
    pass over it on a path an operator walks once by hand.
    """

    text = _checked(statement)
    _require_write(text)
    path = index_path(scope.directory)
    before = _snapshot(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        cursor = connection.execute(text)
        affected = cursor.rowcount
        unreadable = _unreadable_row(connection)
        if unreadable is not None:
            connection.rollback()
            raise SqlRefusal(unreadable)
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise SqlRefusal(f"SQLite refused the statement: {error}") from error
    finally:
        connection.close()
    after = _snapshot(path)
    touched = {
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    }
    return {
        "scope": scope.name,
        "statement": text,
        "rows_affected": max(affected, 0),
        "touched": sorted(touched),
    }


def _checked(statement: str) -> str:
    """Return the statement with its comments removed and its ends trimmed."""

    text = str(statement or "").strip()
    if not text:
        raise SqlRefusal("the statement is empty: write a statement to run")
    if len(text) > MAXIMUM_STATEMENT:
        raise SqlRefusal(
            f"the statement is {len(text)} characters, over the "
            f"{MAXIMUM_STATEMENT} this console runs. Write a smaller one."
        )
    stripped = _without_comments(text)
    if not stripped:
        raise SqlRefusal("the statement is only a comment: write a statement to run")
    if not sqlite3.complete_statement(stripped.rstrip().rstrip(";") + ";"):
        raise SqlRefusal(
            "the statement is incomplete: it needs a statement that SQLite can parse "
            "in full"
        )
    if stripped.rstrip().rstrip(";").count(";") > 0:
        raise SqlRefusal(
            "one statement at a time: this console runs a single statement, because a "
            "write is reindexed once and two of them are not"
        )
    return stripped


def _without_comments(text: str) -> str:
    """Return the statement with its comments taken out.

    A comment is where a second statement hides, so it is removed before the statement
    is counted rather than after.
    """

    without_blocks = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", " ", without_blocks)


def _leading(text: str) -> str:
    """Return the statement's first keyword, lowercased."""

    match = _LEADING.match(_without_comments(text))
    return match.group(1).lower() if match else ""


def _require_read(text: str) -> None:
    """Refuse a read that is not one."""

    lead = _leading(text)
    if lead in _FORBIDDEN:
        raise SqlRefusal(
            f"{lead.upper()} is not something the read path runs. A read is a SELECT; "
            "a write goes through Execute, which reindexes the memory."
        )
    if lead not in _READS:
        raise SqlRefusal(
            f"{lead.upper() or 'that statement'} is not a read. This console runs a "
            "SELECT to read, and INSERT, UPDATE or DELETE on the unit table to write."
        )


def _require_write(text: str) -> None:
    """Refuse a write that reaches past the statements."""

    lead = _leading(text)
    if lead in _FORBIDDEN:
        raise SqlRefusal(
            f"{lead.upper()} is refused by name. This file is the record a memory is "
            "read from, and that changes the file rather than what it remembers."
        )
    if lead not in _WRITES:
        raise SqlRefusal(
            f"{lead.upper() or 'that statement'} is not a write. Execute runs "
            "INSERT, UPDATE or DELETE on the unit table."
        )
    match = _TARGET.match(_without_comments(text))
    target = None
    if match is not None:
        target = (match.group(1) or match.group(2) or "").casefold()
    if target != WRITABLE_TABLE:
        named = target or "an unnamed table"
        raise SqlRefusal(
            f"a write may only touch the {WRITABLE_TABLE} table, and this one reaches "
            f"for {named}. The other tables are the word index, the vectors, and the "
            "record's own bookkeeping, and editing them makes a memory answer "
            "wrongly rather than fail."
        )
    for word in _IDENTIFIER.findall(_without_comments(text)):
        if word.casefold() in PROTECTED_TABLES:
            raise SqlRefusal(
                f"{word} is protected: it is part of what the word index and the "
                "vectors are built from, not a statement."
            )


def _snapshot(path: Path) -> dict[str, str]:
    """Return every statement's identity and its text, for comparing before and after.

    This reads the whole record, which is a cost an operator pays once for a write
    they typed by hand and never pays on the path a tool takes. It is what makes the
    reindex exact rather than a guess about the statement's own ``WHERE`` clause.
    """

    if not path.is_file():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT unit_key, text FROM unit")
        }
    finally:
        connection.close()


def _is_countable(value: object) -> bool:
    """Whether a column the record counts holds something it reads back as a number.

    An absent or empty value is the same absence and reads as zero; words are not. A
    place of ``first`` makes every later read of the row raise a conversion error
    instead of naming anything a reader can act on.
    """

    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    try:
        int(text)
    except ValueError:
        return False
    return True


def _unreadable_row(connection: sqlite3.Connection) -> str | None:
    """Return why a row of the record cannot be read back, or None when every row can.

    One pass over the record, on the connection the write already holds and before it is
    committed. Every row is checked rather than the rows the write's own ``WHERE`` names,
    because a ``WHERE`` that was wrong once will be wrong again.
    """

    rows = connection.execute(
        "SELECT unit_key, text, stamp, recalls FROM unit"
    ).fetchall()
    for unit_key, text, stamp, recalls in rows:
        if not isinstance(text, str):
            return _unreadable(unit_key, "text", text)
        if not _is_countable(stamp):
            return _unreadable(unit_key, "stamp", stamp)
        if not _is_countable(recalls):
            return _unreadable(unit_key, "recalls", recalls)
    return None


def _unreadable(unit_key: object, column: str, value: object) -> str:
    """Return the refusal for one row the record cannot read back."""

    return (
        f"the statement {unit_key!r} would hold {column}={value!r}, which the record "
        "cannot read back: a statement's place and its recall count are whole numbers "
        "and its words are text, so a row like that fails every later read of this "
        "memory. The write was rolled back. Write whole numbers there, or change only "
        "the words."
    )


def drop_vectors(directory: Path, keys: Iterable[str]) -> int:
    """Drop the vectors of the statements named, so they are embedded again.

    This is the whole of the reindex a hand edit needs. A vector describes the text its
    statement held when the model last saw it, so a statement whose text changed has a
    vector that now answers a question nobody asked, and the read would rank it on
    meaning it no longer has.
    """

    wanted = [str(key) for key in keys]
    if not wanted:
        return 0
    path = index_path(directory)
    if not path.is_file():
        return 0
    connection = sqlite3.connect(path)
    try:
        removed = 0
        connection.execute("BEGIN")
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            cursor = connection.execute(
                f"DELETE FROM vector WHERE unit_key IN ({placeholders})", chunk
            )
            removed += max(0, cursor.rowcount)
        connection.commit()
        return removed
    finally:
        connection.close()


__all__ = [
    "MAXIMUM_ROWS",
    "MAXIMUM_STATEMENT",
    "PROTECTED_TABLES",
    "WRITABLE_TABLE",
    "MemoryIndex",
    "SqlRefusal",
    "drop_vectors",
    "sql_execute",
    "sql_query",
]
