"""Building the service this app runs, from the resolved account and its settings.

One factory, so the command line, the app, and a test all reach a service built the
same way. The policy is the settings, the models are named by them, and the cache they
are fetched into is the one they name, so one ``config.toml`` decides what a memory is
matched with and where its model already is on disk.

The project a service treats as current is chosen by the caller, because the four tools
take no project argument: a statement is filed where the session that recorded it is
working, and the choice of session is the caller's, made once when the app is built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AccountConfig, resolve_account
from .models import LocalEmbedder, LocalReranker, model_cache_directory
from .registry import RegisteredProject
from .registry import load as load_projects
from .retrieval import Retrieval, RetrievalSettings
from .service import MemoryService

__all__ = ["build_retrieval", "build_service", "select_projects"]


def build_retrieval(settings: Any) -> Retrieval:
    """Build this process's retrieval from the settings and the local models."""

    policy = RetrievalSettings.from_settings(settings)
    cache = model_cache_directory(settings.cache_root()) if settings else None
    return Retrieval(
        embedder=LocalEmbedder(policy.embedding_model, cache),
        policy=policy,
        reranker=LocalReranker(policy.reranker_model, cache),
    )


def select_projects(
    account: AccountConfig,
    project_root: str | Path | None = None,
    project_name: str | None = None,
) -> list[RegisteredProject]:
    """Return the projects this process serves, in the order it should treat them.

    The first entry is the active project, so the local scope the four tools file into
    is the one the caller named. Naming a project this installation has not recorded is
    refused here rather than at first use, because a session that files a rule about one
    repository into another one is worse than a session that does not start.
    """

    from .registry import RegistryError, resolve

    known = load_projects()
    if project_root is not None:
        wanted = Path(project_root).expanduser().resolve()
        for project in known:
            if project.project_root == wanted:
                return [project, *[item for item in known if item is not project]]
        raise RegistryError(
            f"{wanted} is not a project this installation has recorded. Run "
            "'memory-rag init --project-root <path> --name <name>' for it first; a "
            "project's local memory only exists once the command center knows where "
            "the project is."
        )
    if project_name is not None:
        chosen = resolve(project_name)
        return [chosen, *[item for item in known if item is not chosen]]
    return known


def build_service(
    storage_root: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    project_name: str | None = None,
    config_path: str | Path | None = None,
    overrides: tuple[str, ...] = (),
    account: AccountConfig | None = None,
) -> MemoryService:
    """Return the service this process runs.

    The account's settings decide what every memory is matched with. A project's own
    ``config.toml`` is resolved by the service when it reaches that project, so a
    project layer applies to that project without this factory reading every project's
    file before the first read.
    """

    if account is None:
        account = resolve_account(
            storage_root, config_path=config_path, overrides=overrides
        )
    projects = select_projects(account, project_root, project_name)
    return MemoryService(
        account,
        build_retrieval(account.settings),
        projects=projects,
    )
