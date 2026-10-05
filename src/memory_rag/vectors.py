"""The vectors, in the same file as the index they answer for.

One SQLite file per scope holds three things, and the reason they share a file is
that they are all keyed by the same statement digest and read on the same answer:
the FTS5 word index, the vector of every statement, and a statement's history.
Keeping them together means one file to look at rather than three, one connection
per read rather than two, and one schema version rather than two.

What is still true of the parts: the word index and the vectors are derived from
the statements, so a rebuild restores them by embedding again, while the history
is not derived from anything and survives one. A table whose shape this version
does not write is dropped and rebuilt rather than half-read, which is also what
retires the vectors of statements that have since been reworded.

The record is the table in that file, and these vectors are about the statements
it holds: a vector with no statement behind it is not about anything that exists,
and it is collected rather than left to grow.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

from .index import index_path
from .store import unit_key

__all__ = [
    "VectorStore",
    "cosine_similarity",
    "unit_key",
]


class VectorStore:
    """One scope's vectors: put, look up by key, and search by cosine."""

    def __init__(self, scope_directory: Path, model: str, dimension: int) -> None:
        self.scope_directory = scope_directory
        self.model = model
        self.dimension = dimension
        self.path = index_path(scope_directory)
        self._connection: sqlite3.Connection | None = None

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        from .index import MemoryIndex

        self.scope_directory.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        # The schema is the index's to keep: one file, one version, and a file
        # whose vector table is not the shape this version writes is dropped
        # there rather than half-read here.
        MemoryIndex(self.scope_directory)._ensure_schema(connection)
        self._connection = connection
        return connection

    def close(self) -> None:
        """Close the file, if this object opened it."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> Self:
        self._open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    def put(
        self,
        key: str,
        vector: Sequence[float],
        *,
        stamp: str,
        kind: str,
        expected_text: str | None = None,
    ) -> str | None:
        """Store one statement's vector under its own key, and return that key.

        The key is the record's, not a digest of the text embedded: the text of a
        statement no longer carries its type, so a digest of it would be a different
        identity from the row it belongs to, and the vector would answer for nothing.

        The recorded model and width are checked before the write, so a stale file
        from another model is never appended to silently.
        """

        if len(vector) != self.dimension:
            raise ValueError(
                f"vector has {len(vector)} dimensions, this store holds "
                f"{self.dimension}"
            )
        connection = self._open()
        self._refuse_foreign_model(connection)
        sql = (
            "INSERT OR REPLACE INTO vector"
            "(unit_key, stamp, kind, model, dimension, components) "
        )
        values: tuple[object, ...] = (
            key,
            stamp,
            kind,
            self.model,
            self.dimension,
            _pack(vector),
        )
        if expected_text is None:
            sql += "VALUES (?, ?, ?, ?, ?, ?)"
        else:
            # Check and insert together; an embedding may finish after a record edit.
            sql += (
                "SELECT ?, ?, ?, ?, ?, ? WHERE EXISTS ("
                "SELECT 1 FROM unit WHERE unit_key = ? AND text = ? "
                "AND kind = ? AND CAST(stamp AS TEXT) = ?)"
            )
            values += (key, expected_text, kind, stamp)
        cursor = connection.execute(sql, values)
        connection.commit()
        return key if cursor.rowcount else None

    def _refuse_foreign_model(self, connection: sqlite3.Connection) -> None:
        """Raise if this file holds vectors from a different model."""

        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'embedding_model'"
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('embedding_model', ?)",
                (self.model,),
            )
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('dimension', ?)",
                (str(self.dimension),),
            )
            return
        if row[0] != self.model:
            raise ValueError(
                f"{self.path} holds vectors from {row[0]!r}, not {self.model!r}; "
                "run reindex to rebuild the vectors without deleting the memory record"
            )
        width = connection.execute(
            "SELECT value FROM meta WHERE key = 'dimension'"
        ).fetchone()
        if width is not None and int(width[0]) != self.dimension:
            raise ValueError(
                f"{self.path} holds {width[0]}-dimensional vectors, this model "
                f"produces {self.dimension}"
            )

    def model_is_foreign(self) -> bool:
        """Report whether this file was built by a different model."""

        connection = self._open()
        row = connection.execute(
            "SELECT value FROM meta WHERE key = 'embedding_model'"
        ).fetchone()
        return row is not None and row[0] != self.model

    # -- reading -----------------------------------------------------------

    def count(self) -> int:
        """Return how many vectors this store holds."""

        connection = self._open()
        row = connection.execute("SELECT count(*) FROM vector").fetchone()
        return int(row[0]) if row else 0

    def keys(self) -> set[str]:
        """Return the keys of the units this store can answer from."""

        connection = self._open()
        return {
            str(row[0]) for row in connection.execute("SELECT unit_key FROM vector")
        }

    def has(self, keys: Iterable[str]) -> set[str]:
        """Return which of these keys this store already holds."""

        wanted = list(dict.fromkeys(keys))
        if not wanted:
            return set()
        connection = self._open()
        found: set[str] = set()
        # SQLite's variable limit is high but finite, so the lookup is chunked.
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = connection.execute(
                f"SELECT unit_key FROM vector WHERE unit_key IN ({placeholders})",
                chunk,
            )
            found.update(str(row[0]) for row in rows)
        return found

    def forget(self, keys: Iterable[str]) -> int:
        """Drop these units' vectors, and return how many were dropped.

        Used when a unit's text changed: the old digest is not the unit any more,
        so its vector is not about anything that exists.
        """

        wanted = list(dict.fromkeys(keys))
        if not wanted:
            return 0
        connection = self._open()
        dropped = 0
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            cursor = connection.execute(
                f"DELETE FROM vector WHERE unit_key IN ({placeholders})",
                chunk,
            )
            dropped += int(cursor.rowcount or 0)
        connection.commit()
        return dropped

    def vectors_for(self, keys: Sequence[str]) -> dict[str, tuple[float, ...]]:
        """Return the named units' vectors, and nothing for a unit that has none.

        One query for the whole set rather than one per key, because a read that
        has to collapse a repetition needs every candidate it is holding and not
        a round trip each. A unit whose vector has not been produced yet is simply
        absent, so the caller has one fewer comparison to make rather than a
        failure to handle.
        """

        wanted = list(dict.fromkeys(str(key) for key in keys))
        if not wanted:
            return {}
        connection = self._open()
        found: dict[str, tuple[float, ...]] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = connection.execute(
                "SELECT unit_key, components FROM vector "
                f"WHERE model = ? AND dimension = ? AND unit_key IN ({placeholders})",
                (self.model, self.dimension, *chunk),
            ).fetchall()
            for key, blob in rows:
                found[str(key)] = _unpack(blob)
        return found

    def search(
        self,
        query_vector: Sequence[float],
        pool: int,
        floor: float,
        margin: float,
    ) -> list[tuple[str, float]]:
        """Return the nearest units to a query vector, and their cosine scores.

        An exact scan, not an approximate index: a memory holds thousands of
        units, which is a few milliseconds of arithmetic, and an approximate
        index would have to be opened on every one of a blocking read. Units
        below ``floor`` are dropped, except that a unit within ``margin`` of the
        query's own best candidate is admitted, so a single strong match is not
        thrown away with the rest.
        """

        connection = self._open()
        rows = connection.execute(
            "SELECT unit_key, components FROM vector WHERE model = ? AND dimension = ?",
            (self.model, self.dimension),
        ).fetchall()
        if not rows:
            return []

        scored = [
            (str(key), cosine_similarity(query_vector, _unpack(blob)))
            for key, blob in rows
        ]
        best = max(score for _, score in scored)
        if best < floor:
            return []
        allowed = best - margin
        scored = [item for item in scored if item[1] >= floor or item[1] >= allowed]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(int(pool), 1)]


def _pack(values: Sequence[float]) -> bytes:
    """Return one vector as 32-bit floats, little-endian."""

    import array

    packed = array.array("f", (float(value) for value in values))
    if packed.itemsize != 4:  # pragma: no cover - every platform this runs on
        raise RuntimeError("32-bit floats are required for the vector store")
    return packed.tobytes()


def _unpack(blob: bytes) -> tuple[float, ...]:
    """Return one stored vector as floats."""

    import array

    packed = array.array("f")
    packed.frombytes(blob)
    return tuple(packed)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine of two vectors that are each already at unit length."""

    if len(left) != len(right):
        raise ValueError(f"vector widths differ: {len(left)} and {len(right)}")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))
