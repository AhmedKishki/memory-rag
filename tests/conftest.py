"""Every test hands back the background workers it started.

A service builds one `EmbeddingWorker` per scope, and a worker starts a daemon thread
the first time a statement is queued. A test that does not close its service therefore
leaves that thread running, holding an embedder and its own scope; after a few dozen
tests the run spends its time in GIL contention rather than in the code under test.

So this fixture records the workers a test creates and stops each of them afterwards.
It is a test-hygiene rule, not a product one: the app builds one service per process
and keeps it for the process's life on purpose, which is why `Retrieval.close` exists
rather than being called on a timer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeEmbedder

from memory_rag import retrieval as retrieval_module
from memory_rag.maintenance import EmbeddingWorker
from memory_rag.service import MemoryService


@pytest.fixture(autouse=True)
def _stop_embedding_workers(monkeypatch: pytest.MonkeyPatch) -> object:
    """Stop every embedding worker the test started, whatever the test did."""

    created: list[EmbeddingWorker] = []

    class Tracked(EmbeddingWorker):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(retrieval_module, "EmbeddingWorker", Tracked)
    yield
    for worker in created:
        worker.stop()


@pytest.fixture(autouse=True)
def _account_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every account-level path at one directory per test.

    The account's settings directory, its project record, and its model cache are
    chosen by the platform, so without this a test would read the account it happens
    to run as and register its project in the user's real project list. The override is
    on the environment the shared settings stack reads, which is the one place all of
    them are named.
    """

    from memory_rag import launcher, registry

    root = tmp_path / "account"
    (root / "config").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(root / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))
    monkeypatch.delenv("MEMORY_ULTRARAG_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("ULTRARAG_UI_STORAGE_ROOT", raising=False)
    # The launcher and the registry both resolve their path at call time from the
    # platform, so pointing the environment at a scratch tree moves both.
    assert registry.registry_path().is_relative_to(root)
    assert launcher.state_root(root).is_relative_to(root)
    return root


@pytest.fixture
def storage_root(_account_root: Path) -> Path:
    return _account_root


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """One repository per test, with its own local memory directory created."""

    root = tmp_path / "project"
    (root / ".memory-rag").mkdir(parents=True)
    return root


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def service(storage_root: Path, project_root: Path, embedder: FakeEmbedder):
    """A service over one account and one project, with no model behind it."""

    from memory_rag.config import AccountConfig, global_directory
    from memory_rag.registry import RegisteredProject
    from memory_rag.retrieval import Retrieval, RetrievalSettings

    project = RegisteredProject(
        project_id="test-project",
        project_name="demo",
        project_root=project_root,
        registered_at="",
    )
    account = AccountConfig(
        storage_root=storage_root,
        global_directory=global_directory(storage_root),
    )
    policy = RetrievalSettings.from_settings(None)
    service = MemoryService(
        account,
        Retrieval(embedder=embedder, policy=policy),
        projects=[project],
    )
    yield service
    service.close()
