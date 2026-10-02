"""Where each kind of memory lives, resolved and validated before serving.

Two roots, one behaviour:

* **local memory** is the bound project's own, kept inside the repository under
  ``.memory-rag``, so a project carries its memory with it and no other project
  can see it;
* **global memory** is the user's, kept under the storage root, which defaults to
  this account's home data directory and moves with ``--storage-root`` or
  ``MEMORY_ULTRARAG_STORAGE_ROOT`` — or points at UltraRAG's UI storage tree
  through ``ULTRARAG_UI_STORAGE_ROOT``, so a UltraRAG UI shows the same memory.

The two roots are where memory lives, which is a deployment choice a client
supplies rather than a tunable. Everything else this server reads is a setting,
and the settings are resolved here too: the packaged defaults, the account's
``config.toml`` — the layer that applies globally — the project's own, a file
named on the command line, the environment, and ``--set``, in that order of
precedence.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_path

from .settings import SETTINGS, EffectiveSettings, sources_for
from .settings_layers import SettingsError, resolve_settings

__all__ = [
    "APP_NAME",
    "GLOBAL_MEMORY_DIRNAME",
    "GLOBAL_SCOPE_DIRECTORY",
    "LOCAL_STATE_DIRNAME",
    "PROJECT_ROOT_ENV_VAR",
    "STORAGE_ENV_VAR",
    "ULTRARAG_STORAGE_ENV_VAR",
    "AccountConfig",
    "ConfigurationError",
    "EffectiveSettings",
    "ServerConfig",
    "default_storage_root",
    "global_directory",
    "global_memory_root",
    "resolve_account",
    "resolve_config",
]

APP_NAME = "memory-ultra-rag-mcp"

#: The directory a project's own memory lives in, inside the repository.
LOCAL_STATE_DIRNAME = ".memory-rag"

#: The project this server is bound to, when the host passes it in the environment.
PROJECT_ROOT_ENV_VAR = "MEMORY_ULTRARAG_PROJECT_ROOT"

#: This package's own switch for where a user's global memory lives.
STORAGE_ENV_VAR = "MEMORY_ULTRARAG_STORAGE_ROOT"

#: UltraRAG's own variable for its UI storage tree, honored for interoperability.
ULTRARAG_STORAGE_ENV_VAR = "ULTRARAG_UI_STORAGE_ROOT"

#: The directory inside the storage root that holds global memory, which is
#: UltraRAG's own name for it.
GLOBAL_MEMORY_DIRNAME = "memory"

#: The one directory under that tree that holds the global memory. There is one
#: global memory, not one per user: this server serves the account's memory, and
#: the directory is named as UltraRAG names it when no user is given, so the
#: layout stays upstream's and a UltraRAG UI reads it without being told anything.
GLOBAL_SCOPE_DIRECTORY = "default"


class ConfigurationError(ValueError):
    """Raised when the selected roots cannot hold memory."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """A bound project, the storage root global memory lives under, and its tree.

    ``settings`` is the merged result of the layers, and ``settings_provenance``
    says which layer supplied each value, so ``--print-config`` can show where a
    number came from rather than only what it is.
    """

    project_root: Path
    local_directory: Path
    storage_root: Path
    global_directory: Path
    settings: EffectiveSettings | None = None
    settings_provenance: dict[str, str] | None = None


def default_storage_root() -> Path:
    """Return the home data directory that holds global memory by default.

    The default belongs to the account, not to a project, so every project this
    account opens reads the same one global memory. ``--storage-root``,
    ``MEMORY_ULTRARAG_STORAGE_ROOT``, or ``ULTRARAG_UI_STORAGE_ROOT`` move it.
    """
    return Path(user_data_path(APP_NAME, appauthor=False))


def global_memory_root(storage_root: Path) -> Path:
    """Return the tree that holds global memory, as UltraRAG lays it out."""
    return storage_root / GLOBAL_MEMORY_DIRNAME


def global_directory(storage_root: Path) -> Path:
    """Return the one directory that holds the global memory."""

    return global_memory_root(storage_root) / GLOBAL_SCOPE_DIRECTORY


@dataclass(frozen=True, slots=True)
class AccountConfig:
    """The account's global memory and the settings that apply to every project.

    A command center serves the account, not one repository, so it resolves the
    storage root and the account settings layer once and then reaches a project's own
    layer separately. The two are kept apart deliberately: the account's
    ``config.toml`` is the layer that applies everywhere, and a project's is the
    layer that applies to that project alone.
    """

    storage_root: Path
    global_directory: Path
    settings: EffectiveSettings | None = None
    settings_provenance: dict[str, str] | None = None

    def scope(self) -> dict[str, Path]:
        """Return the one memory this account owns."""

        return {"global": self.global_directory}


def resolve_account(
    storage_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> AccountConfig:
    """Resolve the account's global memory and the layers that apply to it.

    Only the account layers are read, because no project has been named yet. A
    project's own ``config.toml`` is a project layer and belongs to
    :func:`resolve_config`, which the service calls per project.

    Nothing is created here, so resolving an account writes nothing at all.
    """

    storage = _selected_storage_root(storage_root)
    settings, provenance = _resolve_settings_for(
        None, config_path=config_path, overrides=overrides
    )
    return AccountConfig(
        storage_root=storage,
        global_directory=global_directory(storage),
        settings=settings,
        settings_provenance=dict(provenance),
    )


def _resolve_settings_for(
    project: Path | None,
    *,
    config_path: str | Path | None,
    overrides: Sequence[str],
) -> tuple[EffectiveSettings, dict[str, str]]:
    """Return the merged settings for a project, or for the account when it is None."""

    try:
        values, provenance = resolve_settings(
            SETTINGS,
            sources_for(project),
            config_path=config_path,
            overrides=overrides,
        )
    except SettingsError as error:
        raise ConfigurationError(str(error)) from error
    return EffectiveSettings.from_values(values, provenance), provenance


def resolve_config(
    project_root: str | Path,
    storage_root: str | Path | None = None,
    *,
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> ServerConfig:
    """Resolve the settings and validate the two roots before any memory is touched.

    The storage root comes from the argument, then from
    ``MEMORY_ULTRARAG_STORAGE_ROOT``, then from ``ULTRARAG_UI_STORAGE_ROOT``, and
    otherwise from the home data directory. A relative root is resolved against
    the working directory, and an absolute one is used as given.

    The settings are resolved from the same project, so the project's own
    ``config.toml`` is the layer that applies to it and the account's is the layer
    that applies everywhere.

    Nothing is created here. Both directories are made when a memory is first
    touched, so resolving a configuration — which is what ``--print-config`` does —
    writes nothing at all.
    """
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise ConfigurationError(
            f"project root is not a directory: {project}; point it at the "
            "repository whose memory is served"
        )

    storage = _selected_storage_root(storage_root)
    local = project / LOCAL_STATE_DIRNAME

    settings, provenance = _resolve_settings_for(
        project, config_path=config_path, overrides=overrides
    )

    return ServerConfig(
        project_root=project,
        local_directory=local,
        storage_root=storage,
        global_directory=global_directory(storage),
        settings=settings,
        settings_provenance=dict(provenance),
    )


def _selected_storage_root(selected: str | Path | None) -> Path:
    """Return the storage root the argument or the environment selects."""
    for candidate in (
        selected,
        os.environ.get(STORAGE_ENV_VAR),
        os.environ.get(ULTRARAG_STORAGE_ENV_VAR),
    ):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        return Path(text).expanduser().resolve()
    return default_storage_root()
