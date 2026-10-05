"""One scope's memory, as a table, and the rendering of it written on request.

``memory.sqlite3`` is the record. Every statement is a row in ``unit``: its text,
the kind it was filed under, its place in the document, and the four things a
sentence cannot hold — when it was added, how often it has been recalled, and
when it was last recalled. The same table is the FTS5 word index, so a statement
is stored once and found by words without a second copy to keep in step.

``MEMORY.md`` is an **export** of that record: the same statements as plain prose,
newest first, written when something asks for it — ``--export``, or the page on
every request — and never in passing. It exists to be read, diffed, and carried,
and it is never read back as memory: a statement in it that the record does not
hold is one that was removed, and adopting it would undo the removal. A record
with nothing in it adopts the export whole, which is how a memory written by an
earlier version is recovered.

The vectors are in the same file, keyed by the same statement digest, and are the
one derived part a rebuild cannot restore cheaply: they cost one embedding each.
A table whose shape this version does not write is dropped and rebuilt rather than
half-read.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from .store import (
    DEFAULT_KIND,
    Statement,
    StoreError,
    normalise,
    parse_document,
    query_terms,
    read_document,
    render_document,
    unit_key,
)

__all__ = [
    "INDEX_FILENAME",
    "PRODUCTIVE_COLUMNS",
    "SUPERSEDED_FILENAMES",
    "IndexError",
    "MemoryIndex",
    "fts5_available",
    "index_all",
    "index_path",
    "query_terms",
    "rebuild",
    "unit_key",
]

#: The one file inside a scope directory, holding everything derived from the
#: document: the word index, the vectors, and the history no rebuild can produce.
INDEX_FILENAME = "memory.sqlite3"

#: The files an earlier version wrote that are nothing but a copy of the record.
#: They are read for anything the record lacks and then removed, so an upgrade
#: costs nothing and a scope is not left holding a dead database beside a live one.
SUPERSEDED_FILENAMES = ("index.sqlite3", "index-vectors.sqlite3")

#: The file a rendering is written to, and the one a memory of an older version
#: left behind. It is not superseded: it is what a person reads, written on
#: request, and it is removed only when it holds a statement the record has lost.
RENDERED_FILENAME = "MEMORY.md"

#: Bumped when the schema changes, so an older file is read as a previous version
#: rather than as this one. 3 replaced the ``kind`` column, which held one value
#: for every row, with the statement's own type. 4 added a separate history table.
#: 5 brought the vectors in from their own file. 6 made ``unit`` the record
#: itself — it gained the four columns the history table held, plus the normalised
#: text that an exact match compares — and dropped that table, because a statement
#: is one row and not two. 7 renamed that field from ``type`` to ``kind``. 8
#: dropped the two columns nothing read: the normalised copy of the words, which
#: is the statement stored twice, and ``last_recalled_at``, which a read wrote on
#: every statement it returned and no code ever read. What is left on a row is the
#: statement, its identity, and the four facts that do work — see
#: :data:`PRODUCTIVE_COLUMNS`.
SCHEMA_VERSION = "8"

#: Every column of a row except the statement and its identity, with the one thing
#: that reads it. A column earns its place by being asked for, so a column added
#: without naming its reader is a column a memory pays for and nobody uses, and
#: ``tests/test_metadata.py`` fails if one appears. ``unit.text`` is the record and
#: ``unit.unit_key`` is what the vector and the history are found by; neither needs
#: a reader of its own.
PRODUCTIVE_COLUMNS = {
    "kind": "a read narrowed by kind, and the label its identity is computed from",
    "stamp": "the order the document is written in, and the tie-break between memories",
    "added_at": "the recency bonus, and the date a caller is told",
    "recalls": "how often the memory has handed the statement back, which a caller judges by",
}

#: The vector table, and the columns this version writes. A table that does not
#: have exactly these is dropped and rebuilt rather than half-read.
VECTOR_COLUMNS = (
    "unit_key",
    "stamp",
    "kind",
    "model",
    "dimension",
    "components",
)
VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS vector(
    unit_key TEXT PRIMARY KEY,
    stamp TEXT NOT NULL,
    kind TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    components BLOB NOT NULL
);
"""

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


#: The shape every row read out of a file of an earlier one is returned in: the words,
#: the identity, the kind, the place, and the two columns of history. A file that did not
#: have one of them reads with a blank there, because reading only the columns a file
#: happens to have and taking them by position puts the place where the kind belongs and
#: files a statement under a number.
UNIT_SHAPE = ("text", "unit_key", "kind", "stamp", "added_at", "recalls")


def read_unit_rows(path: Path) -> list[tuple[str, ...]] | None:
    """Return the statements a file of an earlier shape holds, or None if unreadable.

    Read-only, tolerant of an older field name, and about the statement table alone: a
    vector table holds no words, so a row of one is not a statement to import and its key
    is not a statement's text. A file that reads and holds nothing is an empty index and
    may be removed; a file that cannot be read at all is something this server does not
    understand, and a caller deciding whether to delete it needs to tell those apart.
    """

    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.DatabaseError:
        return None
    try:
        try:
            columns = {
                str(record[1])
                for record in connection.execute("PRAGMA table_info(unit)")
            }
        except sqlite3.DatabaseError:
            return None
        if "text" not in columns:
            return []

        # An older file called the kind `type`. Every field is named here rather than
        # filtered, so a blank lands where the shape says it does.
        def column(name: str) -> str:
            return name if name != "kind" else ("kind" if "kind" in columns else "type")

        selected = ", ".join(
            column(name) if column(name) in columns else "''" for name in UNIT_SHAPE
        )
        try:
            found = connection.execute(f"SELECT {selected} FROM unit").fetchall()
        except sqlite3.DatabaseError:
            found = []
        return [
            tuple("" if value is None else str(value) for value in row) for row in found
        ]
    finally:
        connection.close()


def _last_position(connection: sqlite3.Connection) -> int:
    """Return the last place this record gives a statement, or -1 when it holds none."""

    row = connection.execute("SELECT max(CAST(stamp AS INTEGER)) FROM unit").fetchone()
    if row is None or row[0] is None:
        return -1
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return -1


def _now() -> str:
    """Return the current time, as text a person can read and a string can order.

    Milliseconds rather than seconds because a burst of statements is written in
    the same second, and the preference for new information is between statements
    recorded moments apart. A second of resolution would make the newest of a
    burst indistinguishable from the first, which is the case it exists for.
    """

    return datetime.now(UTC).isoformat(timespec="milliseconds")


class IndexError(StoreError):
    """Raised when the search index cannot be built or queried."""


def index_path(scope_directory: Path) -> Path:
    """Return the index file for one scope."""

    return scope_directory / INDEX_FILENAME


def fts5_available() -> bool:
    """Report whether this Python's SQLite was built with FTS5.

    The check is a real one: it creates the table this module uses, because a
    build can report a version that still lacks the extension.
    """

    try:
        connection = sqlite3.connect(":memory:")
    except sqlite3.Error:  # pragma: no cover - a Python without sqlite3 at all
        return False
    try:
        connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        return True
    except sqlite3.Error:
        return False
    finally:
        connection.close()


class MemoryIndex:
    """One scope's memory: the record, the search over it, and the document it exports."""

    def __init__(self, scope_directory: Path) -> None:
        self.scope_directory = scope_directory
        self.path = index_path(scope_directory)
        self._connection: sqlite3.Connection | None = None
        #: Statements read out of a file of an older shape, kept until the
        #: document has been asked whether it can supply them instead.
        self._legacy_rows: list[tuple[str, ...]] = []
        #: Files an earlier version wrote, removed the first time this record was
        #: opened after the upgrade. Disclosed, because a file vanishing from a
        #: memory is not something a caller should have to notice on its own.
        self.retired: list[str] = []
        #: Vectors copied out of a version-4 vector file by the last upgrade, and
        #: zero when there was none to copy.
        self.migrated_vectors: int = 0
        #: Whether this object made the record file, or rebuilt its statement table
        #: from a file of an older shape. A record in either state has held no
        #: statement of its own, which is what makes the document beside it a
        #: recovery source rather than a stale copy of what was removed from it.
        self._record_is_new: bool | None = None
        #: Whether the statement table was just rebuilt from a file of an older shape,
        #: set by the schema check during this object's open.
        self._adopted_legacy_shape = False

    # -- lifecycle ---------------------------------------------------------

    def _adopt_legacy_vectors(self, connection: sqlite3.Connection) -> int:
        """Copy a version-4 vector file into this one, then remove it.

        A memory's vectors cost one embedding each, so an upgrade that dropped
        them would be an upgrade that re-pays for the whole memory. The rows are
        keyed by the same statement digest either way, so the move is a copy.
        """

        legacy = self.scope_directory / SUPERSEDED_FILENAMES[1]
        if not legacy.is_file():
            return 0
        moved = 0
        read = False
        try:
            connection.execute("ATTACH DATABASE ? AS legacy", (str(legacy),))
            exists = connection.execute(
                "SELECT count(*) FROM legacy.sqlite_master "
                "WHERE type = 'table' AND name = 'vector'"
            ).fetchone()
            if exists and int(exists[0] or 0):
                read = True
                present = {
                    str(row[1])
                    for row in connection.execute("PRAGMA legacy.table_info(vector)")
                }
                # The columns are named rather than taken in order, and the kind is
                # read from whichever name the file has: a version-5 file also had
                # a `source` column and called the kind `type`. The copy has to work
                # from either shape or the upgrade costs a re-embedding.
                kind_column = "kind" if "kind" in present else "type"
                if {
                    "unit_key",
                    "stamp",
                    kind_column,
                    "model",
                    "dimension",
                    "components",
                } <= present:
                    connection.execute(
                        "INSERT OR IGNORE INTO vector"
                        "(unit_key, stamp, kind, model, dimension, components) "
                        f"SELECT unit_key, stamp, {kind_column}, model, dimension, "
                        "components FROM legacy.vector"
                    )
                    moved = int(
                        connection.execute("SELECT changes()").fetchone()[0] or 0
                    )
            connection.commit()
        except sqlite3.DatabaseError:
            # A file this version cannot read is not worth failing a read over, and
            # it is not worth deleting either: it holds the only copy of embeddings
            # this version cannot rebuild itself. It stays where it is, and the
            # vectors are simply built again.
            read = False
        finally:
            # The detach is a cleanup: a legacy file SQLite already closed itself
            # raises here, and the migration it belonged to is finished either way,
            # so the failure is deliberately swallowed rather than raised over a file
            # that is no longer open.
            with contextlib.suppress(sqlite3.DatabaseError):
                connection.execute("DETACH DATABASE legacy")
        if moved:
            self.migrated_vectors = moved
        if read:
            for suffix in ("", "-wal", "-shm"):
                Path(f"{legacy}{suffix}").unlink(missing_ok=True)
        return moved

    def retire_superseded(self, *, render: bool = False) -> list[str]:
        """Remove the copies a memory does not keep, once the record has their statements.

        A memory used to be two databases and a document, all three read into the record
        when the record became one file and all three left where they were. So a scope
        kept files describing the same memory beside the one that holds it.

        A superseded index file is read first and anything the record lacks is imported,
        which is the one moment a statement that exists only in an older file can be
        rescued. The rendering is not read: it is written from the record, so a statement
        in it that the record does not hold is one that was deleted, and adopting it
        would undo the deletion. :meth:`adopt_document_if_empty` covers the recovery case
        instead, on a record that has never held a statement. The rendering is removed
        only with ``render=True``, and a file that cannot be read is left where it is.

        Nothing is removed until its statements are committed, so a failed commit leaves
        the file holding the only copy.
        """

        connection = self._open()
        known = self.unit_keys()
        held = {statement.text for statement in self.statements()}
        retired: list[str] = []
        removals: list[Path] = []
        # The vector file is not here: `_adopt_legacy_vectors` owns it, copies what
        # it can read out of it, and removes it only when it could. A file this
        # version cannot read is left where it is on either path.
        for name in (SUPERSEDED_FILENAMES[0], RENDERED_FILENAME):
            path = self.scope_directory / name
            if not path.is_file():
                continue
            if name == RENDERED_FILENAME:
                if not render:
                    continue
            else:
                rows = self._readable_rows(path)
                if rows is None:
                    continue
                missing = [
                    row for row in rows if row[0] not in held and row[1] not in known
                ]
                if missing:
                    self._adopt_legacy_unit(connection, missing)
                    # What this file held is now the record's, so the next file is
                    # compared against the record as it now stands rather than as it
                    # stood before this one was read.
                    held = held | {row[0] for row in missing}
                    known = known | {row[1] for row in missing}
            retired.append(name)
            removals.extend(Path(f"{path}{suffix}") for suffix in ("", "-wal", "-shm"))
        # A file is removed only once the statements it held are in the record and
        # SQLite has said so. A commit that fails leaves the record without them while
        # the file still has them, and removing it there destroys the only copy; the
        # failure is raised instead, with both files left as they were.
        connection.commit()
        for candidate in removals:
            candidate.unlink(missing_ok=True)
        if retired:
            self.retired.extend(retired)
        return retired

    def write_render(self, path: Path | None = None) -> Path:
        """Write this record out as prose, and return where it went.

        The only thing in this server that writes a document, and it is asked for
        rather than done on the way past. The file is a rendering: it is not read
        back as memory, and the next export replaces it.
        """

        target = path or (self.scope_directory / RENDERED_FILENAME)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.export(), encoding="utf-8")
        return target

    def _readable_rows(self, path: Path) -> list[tuple[str, str, str, str]] | None:
        """Return what a superseded file holds, or None when it cannot be read.

        A database and a document are read differently and end at the same place: text,
        identity, kind, and place.

        An identity a file recorded is kept. A statement is identified by its kind and its
        words, so recomputing the identity from the words alone drops the kind from the
        digest: the imported row then has an identity no write produces again, and
        recording the same words under the same kind adds a second row. Only a file with
        no identity gets one made for it.
        """

        if path == self.scope_directory / RENDERED_FILENAME:
            text = read_document(self.scope_directory)
            if text is None:
                return None
            return [
                (statement.text, statement.key, statement.kind, str(statement.position))
                for statement in parse_document(text)
            ]
        rows = read_unit_rows(path)
        if rows is None:
            return None
        held: list[tuple[str, str, str, str]] = []
        for text, key, kind, stamp, *_history in rows:
            held.append(
                (
                    text,
                    key or unit_key(f"{kind or DEFAULT_KIND}: {text}"),
                    kind or DEFAULT_KIND,
                    stamp or "0",
                )
            )
        return held

    def _vector_table_is_older_shape(self, connection: sqlite3.Connection) -> bool:
        """Report whether the vector table is not the shape this version writes."""

        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'vector'"
        ).fetchone()
        if row is None:
            return False
        present = {
            str(record[1]) for record in connection.execute("PRAGMA table_info(vector)")
        }
        return present != set(VECTOR_COLUMNS)

    def _open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        connection: sqlite3.Connection | None = None
        try:
            self.scope_directory.mkdir(parents=True, exist_ok=True)
            created = not self.path.is_file()
            connection = sqlite3.connect(self.path)
            connection.execute("PRAGMA journal_mode=WAL")
            self._create_schema(connection)
            empty = connection.execute("SELECT 1 FROM unit LIMIT 1").fetchone() is None
            self._record_is_new = (created or self._adopted_legacy_shape) and empty
            self._connection = connection
        except (OSError, sqlite3.Error) as error:
            if connection is not None:
                connection.close()
            raise IndexError(
                f"the search index at {self.path} could not be opened: {error}. "
                "This server needs a SQLite built with FTS5. Check permissions and "
                "locks; preserve this file because it holds the memory record."
            ) from error
        return self._connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        """Bring one connection's file to this version's shape.

        Exposed for the vector store, which opens the same file and must not
        create or judge a schema of its own.
        """

        self._create_schema(connection)

    def drop_vectors(self) -> int:
        """Drop every vector in this scope, and return how many went."""

        connection = self._open()
        present = connection.execute("SELECT count(*) FROM vector").fetchone()
        count = int(present[0] or 0) if present else 0
        connection.execute("DROP TABLE IF EXISTS vector")
        connection.executescript(VECTOR_SCHEMA)
        connection.commit()
        return count

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        previous_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(unit)")
        }
        self._adopted_legacy_shape = (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'unit'"
            ).fetchone()
            is None
        )
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS unit USING fts5("
            "text, unit_key UNINDEXED, kind UNINDEXED, stamp UNINDEXED, "
            "added_at, recalls)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)"
        )
        connection.executescript(VECTOR_SCHEMA)
        if self._vector_table_is_older_shape(connection):
            # A table from a version whose columns differ is not read. It is
            # dropped, which also collects the vectors of statements that have
            # since been reworded, and the write side rebuilds what is missing.
            connection.execute("DROP TABLE vector")
            connection.executescript(VECTOR_SCHEMA)
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        current = str(row[0]) if row is not None else ""
        if current and current != SCHEMA_VERSION:
            # A record from another shape is not read. The statements in it are
            # the memory, so they are not dropped: what is in the document beside
            # the file is imported instead, which is the only copy a version-5
            # file has. Anything that file held and the document does not is a
            # statement this version cannot see, and the version is bumped again
            # in the release notes when that is a loss.
            self._legacy_rows = self._read_legacy_unit(connection)
            connection.execute("DROP TABLE IF EXISTS unit")
            connection.execute("DELETE FROM meta")
            connection.execute(
                "CREATE VIRTUAL TABLE unit USING fts5("
                "text, unit_key UNINDEXED, kind UNINDEXED, stamp UNINDEXED, "
                "added_at, recalls)"
            )
            # The statements of the replaced table come back from the file it was in,
            # and anything it did not hold can only come from the document beside it,
            # so this is the one upgrade where an existing record file may still adopt
            # one.
            self._adopted_legacy_shape = not {"added_at", "recalls"} <= previous_columns
        if current != SCHEMA_VERSION:
            # Only when it differs, so a read on a current file takes no write
            # lock: a read that had to write would fail against a scope another
            # process is writing, and a read has no business doing that.
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
        self._adopt_legacy_vectors(connection)
        if self._legacy_rows:
            # This file's own rows, not the document beside it. An export holds
            # the statements and nothing else — no kinds, no dates, no counts — so
            # restoring from it would quietly file every statement as `ITEM` and
            # lose the history. The document is the fallback for a file that has no
            # rows of its own, which is a genuine recovery.
            self._adopt_legacy_unit(connection, self._legacy_rows)
            self._legacy_rows = []
        connection.commit()

    def _read_legacy_unit(
        self, connection: sqlite3.Connection
    ) -> list[tuple[str, ...]]:
        """Return this file's statements as they were, in the shape it held them."""

        columns = {
            str(record[1]) for record in connection.execute("PRAGMA table_info(unit)")
        }
        if "text" not in columns:
            return []
        # A file from before 7 held the field as `type`; it is the same value
        # under the name it is now read as, so either column is accepted. The
        # history columns came with the record itself, so they are read where they
        # are there and defaulted where an older file has none.
        source = "kind" if "kind" in columns else "type"
        wanted = [
            name if name in columns else "NULL"
            for name in ("text", "unit_key", source, "stamp", "added_at", "recalls")
        ]
        query = ", ".join(wanted)
        try:
            rows = connection.execute(f"SELECT {query} FROM unit").fetchall()
        except sqlite3.DatabaseError:
            return []
        return [
            tuple("" if value is None else str(value) for value in row) for row in rows
        ]

    def _adopt_legacy_unit(
        self,
        connection: sqlite3.Connection,
        rows: list[tuple[str, ...]],
    ) -> int:
        """Restore this file's statements, with the history they carried.

        A file of an older shape is a better source than the document beside it,
        because an export holds the statements and nothing else. The kinds, the
        dates, and the counts come back with the words; a file that had no history
        columns gets an empty one, which is the truth about it.

        Identity is checked here rather than by ``INSERT OR IGNORE``, because the
        table this writes into is an FTS5 virtual table and has no constraint to
        ignore against: a statement the record already holds, or one the file holds
        twice, would arrive as a second row of the same statement.

        Statements read beside a record that already holds some are placed below the
        record's own, in the order the file listed them, because the record cannot
        place them and the file's own positions would put two of them at the same
        place in the document. A file read into an empty record keeps its positions,
        which is the one case where they are the only ordering there is.
        """

        moved = 0
        known = {str(row[0]) for row in connection.execute("SELECT unit_key FROM unit")}
        placed = _last_position(connection) + 1 if known else None
        for row in rows:
            text, key, kind, stamp = row[0], row[1], row[2], row[3]
            kind = kind or DEFAULT_KIND
            key = key or unit_key(f"{kind}: {text}")
            stamp = stamp or "0"
            if key in known:
                continue
            known.add(key)
            history = row[4:]
            connection.execute(
                "INSERT OR IGNORE INTO unit"
                "(text, unit_key, kind, stamp, added_at, recalls) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    text,
                    key,
                    kind,
                    str(placed) if placed is not None else stamp,
                    history[0] if len(history) > 0 and history[0] else None,
                    int(history[1]) if len(history) > 1 and history[1] else 0,
                ),
            )
            if placed is not None:
                placed += 1
            moved += 1
        return moved

    def close(self) -> None:
        """Close the index file, settling whatever this object still owes it.

        SQLite holds a transaction open until something commits it and discards an open
        one on close, so the object's lifetime owns its transaction: what it wrote is
        durable when it closes, and a read-only object commits nothing.

        A failed commit is raised and the statements it would have saved are rolled back,
        because a caller told a memory was written when it was not has no other way to
        find out.
        """

        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.commit()
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                connection.rollback()
            raise
        finally:
            connection.close()

    def __enter__(self) -> Self:
        self._open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Commit on a clean exit, roll back on a failed one, and release either way.

        A failed operation commits nothing. The rows it staged are half of an answer the
        caller was told did not happen, so a caller that rolls its own work back and
        leaves this block would otherwise find this block's commit turning the failure
        into a partial record.
        """

        if exc_type is None:
            self.close()
            return
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            with contextlib.suppress(sqlite3.Error):
                connection.rollback()
        finally:
            connection.close()

    # -- the record ------------------------------------------------------------

    def statements(self) -> list[Statement]:
        """Return every statement in this scope, newest first, as the record holds it.

        This is the memory. Nothing reads the document to know what is in it, and
        nothing derives the row list from anything else.
        """

        connection = self._open()
        rows = connection.execute(
            "SELECT text, unit_key, kind, stamp, added_at, recalls "
            "FROM unit ORDER BY CAST(stamp AS INTEGER)"
        ).fetchall()
        return [
            Statement(
                kind=str(kind),
                text=str(text),
                key=str(key),
                position=int(stamp or 0),
                added_at=added_at or None,
                recalls=int(recalls or 0),
            )
            for text, key, kind, stamp, added_at, recalls in rows
        ]

    def statements_by_kind(self, kind: str) -> set[str]:
        """Return the identity of every statement filed under this kind."""

        connection = self._open()
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT unit_key FROM unit WHERE kind = ?", (kind,)
            )
        }

    def insert(
        self,
        text: str,
        kind: str,
        *,
        when: str | None = None,
        replace_kind: str | None = None,
    ) -> tuple[str, int]:
        """Put one statement at the top of the record, and return its key and what it replaced.

        A statement's place is its ordinal counting from the top, so inserting
        renumbers the ones below it. Renumbering is one update per statement
        rather than a rewrite of the file, and it is the price of a document that
        reads newest first.

        ``replace_kind`` removes every statement already filed under it first, which
        is how a handoff replaces the previous one without a second call and
        without the caller having to know a handoff exists.
        """

        connection = self._open()
        # Re-recording the same words is one statement, not two, so the first date
        # it was recorded is the date it keeps: how long something has been known
        # is not reset by a caller saying it again.
        previous = connection.execute(
            "SELECT added_at, recalls FROM unit WHERE unit_key = ?",
            (unit_key(f"{kind}: {text}"),),
        ).fetchone()
        replaced = 0
        if replace_kind is not None:
            cursor = connection.execute(
                "DELETE FROM unit WHERE kind = ?", (replace_kind,)
            )
            replaced = int(cursor.rowcount or 0)
        connection.execute("UPDATE unit SET stamp = CAST(stamp AS INTEGER) + 1")
        key = unit_key(f"{kind}: {text}")
        # FTS5 has no unique constraint to enforce identity on, so it is enforced
        # here: without the delete, recording the same words twice would leave two
        # rows of one statement rather than one row and one count.
        connection.execute("DELETE FROM unit WHERE unit_key = ?", (key,))
        connection.execute(
            "INSERT INTO unit"
            "(text, unit_key, kind, stamp, added_at, recalls) "
            "VALUES (?, ?, ?, 0, ?, ?)",
            (
                text,
                key,
                kind,
                when or (previous[0] if previous else None) or _now(),
                int(previous[1] or 0) if previous else 0,
            ),
        )
        connection.commit()
        self._record_is_new = False
        return key, replaced

    def matches(self, text: str) -> list[Statement]:
        """Return the statements whose text is exactly this, and remove none.

        The question and the removal are separate, because a caller searching two
        memories has to see every match before it removes any of them: the whole
        point of an exact match is that nobody loses two statements to one vague
        question.
        """

        wanted = normalise(text)
        return [
            statement
            for statement in self.statements()
            if statement.normalized == wanted
        ]

    def drop(self, key: str) -> int:
        """Remove one statement by its identity, and return whether it went."""

        connection = self._open()
        cursor = connection.execute("DELETE FROM unit WHERE unit_key = ?", (key,))
        connection.commit()
        self._record_is_new = False
        return int(cursor.rowcount or 0)

    def forget(self, normalized: str) -> list[Statement]:
        """Remove the statements whose text is exactly this, and return them.

        Nothing is removed unless it matched exactly, and a text that matches more
        than one statement removes none of them: the whole point of an exact match
        is that a caller cannot lose two statements by asking vaguely.
        """

        found = [
            statement
            for statement in self.statements()
            if statement.normalized == normalized
        ]
        if len(found) == 1:
            self.drop(found[0].key)
        return found

    def replace_all(self, statements: list[Statement]) -> dict[str, int]:
        """Make the record hold exactly these statements, in this order.

        What a person editing the document means. A statement whose words are
        unchanged keeps its kind, its date, and its count, because it is the same
        statement however the file around it was arranged: a rendering carries no
        kinds, so the statement the record filed under `NOTE` and the page sent
        back as plain prose is that statement rather than a new one filed as
        `ITEM`. A kind the incoming statement names is honoured, so a document
        written with `KIND: ` prefixes does re-file a statement. One that is new
        is dated now, and one that is gone is dropped with its history, which is
        the point of the edit. A statement the document holds twice is stored
        once, because the table has no constraint to ignore against. The number
        of each is returned, so a caller can say what the edit did rather than
        only that it was saved.
        """

        connection = self._open()
        before: dict[str, dict[str, object]] = {}
        by_words: dict[str, str] = {}
        for key, text, kind, added_at, recalls in connection.execute(
            "SELECT unit_key, text, kind, added_at, recalls FROM unit"
        ):
            key = str(key)
            before[key] = {
                "kind": str(kind),
                "added_at": added_at or None,
                "recalls": int(recalls or 0),
            }
            by_words.setdefault(normalise(str(text)), key)
        settled: list[tuple[str, Statement]] = []
        keep: set[str] = set()
        for statement in statements:
            key = statement.key
            if key not in before and statement.kind == DEFAULT_KIND:
                held = by_words.get(normalise(statement.text))
                if held is not None:
                    key = held
            if key in keep:
                continue
            keep.add(key)
            settled.append((key, statement))
        removed = [key for key in before if key not in keep]
        for key in removed:
            connection.execute("DELETE FROM unit WHERE unit_key = ?", (key,))
        for position, (key, statement) in enumerate(settled):
            facts = before.get(key, {})
            connection.execute("DELETE FROM unit WHERE unit_key = ?", (key,))
            connection.execute(
                "INSERT INTO unit"
                "(text, unit_key, kind, stamp, added_at, recalls) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    statement.text,
                    key,
                    str(facts.get("kind") or statement.kind),
                    str(position),
                    facts.get("added_at") or _now(),
                    int(facts.get("recalls") or 0),
                ),
            )
        connection.commit()
        self._record_is_new = False
        unchanged = len(before) - len(removed)
        return {
            "kept": unchanged,
            "added": len(settled) - unchanged,
            "removed": len(removed),
            "vectors_removed": self.collect_vectors(),
        }

    def collect_vectors(self) -> int:
        """Drop the vectors of statements the record no longer holds.

        The record is the memory, so a vector whose statement is gone is not about
        anything that exists, and the file would otherwise grow with them. A set
        difference over keys rather than a pass, and it is the reindex that runs it
        because deciding it is a whole-scope call.
        """

        connection = self._open()
        keys = {
            str(row[0]) for row in connection.execute("SELECT unit_key FROM vector")
        }
        stale = [key for key in keys if key not in self.unit_keys()]
        if not stale:
            return 0
        connection.execute("BEGIN")
        for start in range(0, len(stale), 500):
            chunk = stale[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            connection.execute(
                f"DELETE FROM vector WHERE unit_key IN ({placeholders})", chunk
            )
        connection.commit()
        return len(stale)

    def export_digest(self) -> str:
        """Return the digest of this record rendered as a document.

        What the page is handed, so it can tell whether the memory changed while
        the page was open rather than overwriting somebody else's edit.
        """

        return hashlib.sha256(self.export().encode("utf-8")).hexdigest()

    def export(self) -> str:
        """Return the document this record renders as: prose, newest first.

        Rendered, not written. The record is the memory and this is how it reads,
        which the browser view shows and a person can be shown.
        """

        return render_document(self.statements())

    def _adopt_document(self, connection: sqlite3.Connection, document: str) -> bool:
        """Import the statements a document holds, and return whether it held any.

        This is the upgrade path and the recovery path both: a file from a version
        that kept the document as the record, read once and then written back out
        by the first export. It is not a hand edit, because it runs only when the
        record has nothing in it.

        A statement the file holds twice is imported once. The table has no
        constraint to ignore against, so the check is here: a memory that answers
        one question twice is a memory nobody can be sure of, and a file written by
        hand is the place that duplication comes from.
        """

        if connection.execute("SELECT count(*) FROM unit").fetchone()[0]:
            return False
        adopted = False
        seen: set[str] = set()
        for parsed in parse_document(document):
            if parsed.key in seen:
                continue
            seen.add(parsed.key)
            connection.execute(
                "INSERT OR IGNORE INTO unit"
                "(text, unit_key, kind, stamp, added_at, recalls) "
                "VALUES (?, ?, ?, ?, NULL, 0)",
                (
                    parsed.text,
                    parsed.key,
                    parsed.kind,
                    str(parsed.position),
                ),
            )
            adopted = True
        return adopted

    def adopt_document_if_empty(self) -> int:
        """Recover legacy prose only into a new or migrated empty statement table.

        Successful record writes end eligibility, including writes that empty the table.
        """

        connection = self._open()
        if self._record_is_new is not True or self.count_units():
            return 0
        text = read_document(self.scope_directory)
        if text is None:
            return 0
        moved = self._adopt_document(connection, text)
        connection.commit()
        self._record_is_new = False
        if not moved:
            return 0
        return int(connection.execute("SELECT count(*) FROM unit").fetchone()[0] or 0)

    # -- a statement's history ------------------------------------------------

    def note_recalled(self, keys: Sequence[str]) -> None:
        """Count that these statements were handed to a caller.

        One ``UPDATE`` per key inside the connection the read already holds, so
        the work is proportional to the answer rather than to the memory, and
        SQLite's own transaction makes it atomic. A count is all this writes: a
        read is not a time of writing, so a statement keeps the date of the write
        that recorded it and nothing else.
        """

        wanted = [key for key in dict.fromkeys(keys) if key]
        if not wanted:
            return
        connection = self._open()
        for key in wanted:
            connection.execute(
                "UPDATE unit SET recalls = CAST(recalls AS INTEGER) + 1 "
                "WHERE unit_key = ?",
                (key,),
            )
        connection.commit()

    def history(self, keys: Sequence[str]) -> dict[str, dict[str, object]]:
        """Return these statements' histories, and nothing for the rest."""

        wanted = list(dict.fromkeys(keys))
        if not wanted:
            return {}
        connection = self._open()
        found: dict[str, dict[str, object]] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = connection.execute(
                f"SELECT unit_key, added_at, recalls FROM unit "
                f"WHERE unit_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for key, added_at, recalls in rows:
                found[str(key)] = {
                    "added_at": added_at or None,
                    "recalls": int(recalls or 0),
                }
        return found

    def count_units(self) -> int:
        """Return how many units this index holds, without materialising them.

        A disclosure number, so it is two counts rather than a walk: the read
        path must not pay per unit to say how many are waiting.
        """

        connection = self._open()
        row = connection.execute("SELECT count(*) FROM unit").fetchone()
        return int(row[0]) if row else 0

    def unit_keys(self) -> set[str]:
        """Return the identity of every statement this record holds."""

        connection = self._open()
        return {str(row[0]) for row in connection.execute("SELECT unit_key FROM unit")}

    def units_by_key(self, keys: Iterable[str]) -> dict[str, dict[str, str]]:
        """Return the units named by these keys, with the text each one holds."""

        wanted = list(dict.fromkeys(keys))
        if not wanted:
            return {}
        connection = self._open()
        found: dict[str, dict[str, str]] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = connection.execute(
                "SELECT unit_key, text, kind, stamp FROM unit "
                f"WHERE unit_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for key, text_value, kind, stamp in rows:
                found[str(key)] = {
                    "text": str(text_value),
                    "kind": str(kind),
                    "stamp": str(stamp),
                }
        return found

    def search(
        self,
        query: str,
        limit: int,
        kind: str | None = None,
    ) -> tuple[list[dict[str, object]], str]:
        """Return the statements matching a query, and how they were matched.

        The query's words are required to be present together first, which is
        what a question like "where does ref000137 keep the draft" means and what
        an index answers without visiting the rest of the memory. When nothing
        holds every word, the rarer words are tried on their own, because an
        over-strict conjunction returning nothing is worse than a broader one.

        A ``kind`` narrows the answer to one category. The kind is a column
        rather than a word in the text, so a caller that wants a category asks for
        it and a query that merely names one finds nothing — which is the honest
        answer, because the text does not contain it.
        """

        terms = [term for term in query_terms(query) if term]
        if not terms:
            return [], "no-terms"

        connection = self._open()
        rows = self._match(
            connection,
            " ".join(f'"{term}"' for term in terms),
            limit,
            kind,
        )
        mode = "all-words"
        if not rows:
            rarer = self._rarest(connection, terms)[:2]
            if rarer:
                rows = self._match(
                    connection,
                    " OR ".join(f'"{term}"' for term in rarer),
                    limit,
                    kind,
                )
                mode = "rarest-words" if rows else "all-words"
        return rows, mode

    def _match(
        self,
        connection: sqlite3.Connection,
        expression: str,
        limit: int,
        kind: str | None = None,
    ) -> list[dict[str, object]]:
        """Run one FTS5 query, ordered by the index's own ranking.

        A kind narrows the query rather than being part of it. The kind is not in
        the text — it is a column, and a file of tagged lines is not what a memory
        should be — so ``PLAN`` as a word finds nothing and the column is what a
        caller asking for a category gets.
        """

        clause = "unit MATCH ?"
        parameters: list[object] = [expression]
        if kind is not None:
            clause += " AND kind = ?"
            parameters.append(kind)
        try:
            cursor = connection.execute(
                f"SELECT text, unit_key, kind, stamp FROM unit "
                f"WHERE {clause} ORDER BY rank LIMIT ?",
                (*parameters, max(int(limit), 1) + 1),
            )
        except sqlite3.Error as error:
            raise IndexError(
                f"the query could not be run against the index: {error}"
            ) from error
        found = cursor.fetchall()
        # One row past the limit is how the caller learns there was more.
        return [
            {
                "text": text,
                "unit_key": key,
                "kind": kind,
                "stamp": stamp,
            }
            for text, key, kind, stamp in found
        ]

    def _rarest(
        self, connection: sqlite3.Connection, terms: Iterable[str]
    ) -> list[str]:
        """Return the query's words that appear in the fewest units, present ones only.

        A word that occurs in a handful of units is the one worth searching on
        when a conjunction finds nothing; a word in every unit finds everything,
        and a word in no unit finds nothing at all, so both are left out.
        """

        counts: dict[str, int] = {}
        for term in terms:
            row = connection.execute(
                "SELECT count(*) FROM unit WHERE unit MATCH ?", (f'"{term}"',)
            ).fetchone()
            count = int(row[0]) if row else 0
            if count:
                counts[term] = count
        return sorted(counts, key=lambda term: counts[term])


def rebuild(scope_directory: Path) -> dict[str, object]:
    """Settle one scope's derived state, and report what it now holds.

    A record is not a cache, so nothing here deletes it: the statements, the kinds, the
    stamps, the dates, and the recall counts *are* the file, and the only derived parts of
    it are the vectors and the rendering — the word index is the statement table itself,
    because it is an FTS5 table.

    So a rebuild drops the vectors of statements the record no longer holds and writes
    the rendering again, and lets the caller embed what is missing. Retiring the rendering
    removes the file, so the write has to follow it. An export is adopted first when the
    record has never held a statement, which is the one case where it holds one the
    record does not.
    """

    with MemoryIndex(scope_directory) as index:
        index.adopt_document_if_empty()
        vectors_removed = index.collect_vectors()
        index.retire_superseded(render=True)
        rendered = index.write_render()
        return {
            "statements": index.count_units(),
            "vectors_removed": vectors_removed,
            "rendered": str(rendered),
        }


def index_all(
    scope_directories: Sequence[Path],
) -> dict[str, dict[str, object]]:  # pragma: no cover
    """Settle several scopes, for a maintenance caller."""

    return {str(scope): rebuild(scope) for scope in scope_directories}
