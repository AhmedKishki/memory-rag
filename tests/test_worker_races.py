"""Derived vectors must match the record when embedding finishes."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from memory_rag import maintenance
from memory_rag.index import MemoryIndex
from memory_rag.maintenance import EmbeddingWorker, PendingUnit
from memory_rag.models import ModelIdentity
from memory_rag.vectors import VectorStore


class Embedder:
    identity = ModelIdentity("embedding", "worker-race-test", 2)

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]


class PausedEmbedder(Embedder):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.entered.set()
        assert self.release.wait(5), "test did not release embedding"
        return super().embed_documents(texts)


def pending(directory: Path) -> PendingUnit:
    with MemoryIndex(directory) as index:
        key, _ = index.insert("Prefer spaces", "RULE")
        row = index.units_by_key([key])[key]
    return PendingUnit(key, str(row["text"]), str(row["kind"]), str(row["stamp"]))


@pytest.mark.parametrize(
    ("column", "value"),
    [(None, None), ("text", "Prefer tabs"), ("kind", "PREFERENCE"), ("stamp", "99")],
)
def test_embedding_does_not_outlive_its_record_snapshot(
    tmp_path: Path, column: str | None, value: str | None
) -> None:
    unit = pending(tmp_path)
    embedder = PausedEmbedder()
    worker = EmbeddingWorker(embedder, tmp_path)
    try:
        worker.submit(unit)
        assert embedder.entered.wait(5)
        with MemoryIndex(tmp_path) as index:
            if column is None:
                index.drop(unit.key)
            else:
                index._open().execute(
                    f"UPDATE unit SET {column} = ? WHERE unit_key = ?",
                    (value, unit.key),
                )
        embedder.release.set()
        assert worker.drain(5)
        with VectorStore(tmp_path, embedder.identity.name, 2) as store:
            assert store.count() == 0
    finally:
        embedder.release.set()
        worker.stop()


def test_repeated_storage_failures_are_reported_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    unit = pending(tmp_path)
    original = VectorStore.put
    attempts = 0

    def interrupted(
        self: VectorStore,
        key: str,
        vector: Sequence[float],
        *,
        stamp: str,
        kind: str,
        expected_text: str | None = None,
    ) -> str | None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("lookup write unavailable")
        return original(
            self, key, vector, stamp=stamp, kind=kind, expected_text=expected_text
        )

    monkeypatch.setattr(VectorStore, "put", interrupted)
    monkeypatch.setattr(maintenance, "RETRY_SECONDS", 0.0)
    worker = EmbeddingWorker(Embedder(), tmp_path)
    try:
        worker.submit(unit)
        assert worker.drain(5)
        assert attempts == 3
        assert caplog.text.count("stays pending") == 1
        with VectorStore(tmp_path, Embedder.identity.name, 2) as store:
            assert store.has([unit.key]) == {unit.key}
    finally:
        worker.stop()
