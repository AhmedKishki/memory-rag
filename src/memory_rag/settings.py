"""Every tunable this server reads, as one registry over a shared layer stack.

The settings are declared in a table here and shipped as ``default.toml`` beside
this module, so a value that exists in code and a value a file may set cannot
drift apart: a layer may only set a key this registry declares, and a key the
registry does not declare is an error rather than a silent default.

The layers, lowest precedence first:

1. the packaged ``default.toml``, which holds every setting;
2. the user's ``config.toml``, in the account's config directory — the one that
   applies globally, since a memory is shared by every project on the account;
3. the project's own ``config.toml``, inside the project's own state, for a
   project that wants to differ from the account;
4. a file named with ``--config``;
5. ``MEMORY_ULTRARAG_*`` environment variables, one declared per setting;
6. ``--set key=value`` on the command line.

Each layer names only the keys it changes, and every effective value reports the
layer that supplied it, so ``--print-config`` can say where a number came from
rather than only what it is.

The stack itself, the registry's ``Setting`` type, the coercion every layer
shares, the provenance, and the three path helpers live in ``settings_layers``
beside this module. What is declared here is this app's own vocabulary: the
keys, their types and bounds, the packaged default, and the effective settings
the code reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import available_rerankers
from .settings_layers import (
    Setting,
    SettingsError,
    SettingsSources,
    default_config_path,
    project_config_path,
    user_config_path,
)

__all__ = [
    "SETTINGS",
    "SETTINGS_BY_KEY",
    "EffectiveSettings",
    "Setting",
    "SettingsError",
    "SettingsSources",
    "sources_for",
]

#: The names this server resolves its own layers by, which the shared stack takes
#: as arguments. The account directory and the project file are this server's
#: choice: another server in this collection names its own.
USER_CONFIG_DIRECTORY = "memory-ultra-rag-mcp"
PROJECT_CONFIG_RELATIVE = Path(".memory-rag") / "config.toml"

#: Every setting below carries one of this server's own three classes: `identity`,
#: which decides what a memory is made of and is written beside the statements it
#: produced; `runtime`, which chooses where something lives; and `retrieval`, which
#: decides what a read admits and in what order. The shared stack carries the word
#: through untouched, because that classification is this server's to make.


SETTINGS: tuple[Setting, ...] = (
    Setting(
        "runtime.model_cache_root",
        "model_cache_root",
        str,
        "runtime",
        "Shared FastEmbed model cache.",
        env="MEMORY_ULTRARAG_MODEL_CACHE_ROOT",
        empty="",
    ),
    Setting(
        "retrieval.cosine_floor",
        "cosine_floor",
        float,
        "retrieval",
        "The cosine a dense candidate must clear to be admitted at all.",
        env="MEMORY_ULTRARAG_RETRIEVAL_COSINE_FLOOR",
        minimum=-1.0,
        maximum=1.0,
    ),
    Setting(
        "retrieval.relative_margin",
        "relative_margin",
        float,
        "retrieval",
        "How far under its own best a candidate may be and still be admitted.",
        env="MEMORY_ULTRARAG_RETRIEVAL_RELATIVE_MARGIN",
        minimum=0.0,
        maximum=1.0,
    ),
    Setting(
        "retrieval.bm25_weight",
        "bm25_weight",
        float,
        "retrieval",
        "The weight the words contribute to a fused score.",
        env="MEMORY_ULTRARAG_RETRIEVAL_BM25_WEIGHT",
        minimum=0.0,
    ),
    Setting(
        "retrieval.dense_weight",
        "dense_weight",
        float,
        "retrieval",
        "The weight the meaning contributes to a fused score.",
        env="MEMORY_ULTRARAG_RETRIEVAL_DENSE_WEIGHT",
        minimum=0.0,
    ),
    Setting(
        "retrieval.rrf_k",
        "rrf_k",
        int,
        "retrieval",
        "The reciprocal rank fusion constant; larger flattens the sides' ranks.",
        env="MEMORY_ULTRARAG_RETRIEVAL_RRF_K",
        minimum=1.0,
    ),
    Setting(
        "retrieval.recency_bonus",
        "recency_bonus",
        float,
        "retrieval",
        "What a statement is worth for being the newest of those that matched.",
        env="MEMORY_ULTRARAG_RETRIEVAL_RECENCY_BONUS",
        minimum=0.0,
        maximum=1.0,
    ),
    Setting(
        "retrieval.duplicate_cosine",
        "duplicate_cosine",
        float,
        "retrieval",
        "How alike two statements may be before a read keeps only the better of them.",
        env="MEMORY_ULTRARAG_RETRIEVAL_DUPLICATE_COSINE",
        minimum=-1.0,
        maximum=2.0,
    ),
    Setting(
        "retrieval.pool_depth",
        "pool_depth",
        int,
        "retrieval",
        "How many candidates each side contributes before fusion.",
        env="MEMORY_ULTRARAG_RETRIEVAL_POOL_DEPTH",
        minimum=1.0,
    ),
    Setting(
        "retrieval.rerank_depth",
        "rerank_depth",
        int,
        "retrieval",
        "How many fused candidates the cross-encoder reads. One reranks every read; "
        "a larger number reranks more of them, at about 5 ms a candidate.",
        env="MEMORY_ULTRARAG_RETRIEVAL_RERANK_DEPTH",
        minimum=1.0,
    ),
    Setting(
        "dense.embedding_model",
        "embedding_model",
        str,
        "identity",
        "The local model a scope's vectors are built with.",
        env="MEMORY_ULTRARAG_EMBEDDING_MODEL",
    ),
    Setting(
        "dense.reranker_model",
        "reranker_model",
        str,
        "identity",
        "The local cross-encoder that orders every read's candidates. Every read is "
        "reranked, so the value must name one of the pinned models and there is no "
        "empty one.",
        env="MEMORY_ULTRARAG_RERANKER_MODEL",
        choices=available_rerankers(),
    ),
)

SETTINGS_BY_KEY = {setting.key: setting for setting in SETTINGS}
SETTINGS_BY_FIELD = {setting.field: setting for setting in SETTINGS}
SETTINGS_SECTIONS = tuple(dict.fromkeys(s.key.split(".")[0] for s in SETTINGS))


def sources_for(project_root: str | Path | None) -> SettingsSources:
    """Return where this app's file layers live, for one project or for the account.

    The three names are this app's own: the account directory that applies to every
    project, the project file inside `.memory-rag`, and the packaged default that ships
    beside this module. The shared stack reads them and nothing else.

    ``None`` names the account alone, which is the account's settings with no project
    layer: a command center resolves its own settings before it has reached any
    project, and a project layer that applied then would be a project's settings
    deciding the account's.
    """

    if project_root is None:
        return SettingsSources(
            default_file=default_config_path(__file__),
            user_config=user_config_path(USER_CONFIG_DIRECTORY),
            project_root=None,
            project_config=None,
        )
    project = Path(project_root)
    return SettingsSources(
        default_file=default_config_path(__file__),
        user_config=user_config_path(USER_CONFIG_DIRECTORY),
        project_root=project,
        project_config=project_config_path(project, PROJECT_CONFIG_RELATIVE),
    )


@dataclass(frozen=True, slots=True)
class EffectiveSettings:
    """The values in force for one process, and where each came from."""

    runtime: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    dense: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_values(
        cls, values: Mapping[str, Any], provenance: Mapping[str, str]
    ) -> EffectiveSettings:
        """Return the settings, grouped the way the file is."""

        grouped: dict[str, dict[str, Any]] = {
            section: {} for section in SETTINGS_SECTIONS
        }
        for setting in SETTINGS:
            grouped[setting.key.split(".")[0]][setting.field] = values[setting.field]
        return cls(provenance=dict(provenance), **grouped)

    def __getattr__(self, name: str) -> Any:
        """Return a setting by field name, so callers read ``settings.rerank_depth``."""

        for section in SETTINGS_SECTIONS:
            found = getattr(self, section, None)
            if isinstance(found, dict) and name in found:
                return found[name]
        raise AttributeError(name)

    def as_values(self) -> dict[str, Any]:
        """Return the effective values keyed by field, the shape the stack reads."""

        return {setting.field: getattr(self, setting.field) for setting in SETTINGS}

    def cache_root(self) -> str:
        """Return the configured shared cache directory, or "" for this package's own.

        Named apart from the setting it carries: a field is read straight off an
        ``EffectiveSettings``, so a method of the same name would answer instead of
        the value.
        """

        return str(self.runtime["model_cache_root"])
