"""The two model seams, and the local models that fill them.

Nothing in ``store.py`` or ``index.py`` imports a model library, because the
store is a record in plain Markdown and the indexes are derived from it. The two
protocols here are what retrieval is written against, so a stronger local model
is a configuration change, and a hosted backend later is additive rather than a
rewrite.

Both implementations are **local**: the only shipped ones run on this machine,
on CPU, from a cache this package owns. Nothing here opens a network
connection after the model has been fetched once, and no API is used.

``Embedder`` has two methods rather than one because a query and a stored unit
are not the same thing: bge's models are trained with a search instruction in
front of a query and without it in front of a document, and a model that
ignores that distinction answers noticeably worse.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from platformdirs import user_cache_path

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "Embedder",
    "LocalEmbedder",
    "LocalReranker",
    "ModelError",
    "ModelIdentity",
    "Reranker",
    "available_embedders",
    "available_rerankers",
    "model_cache_directory",
]

#: The model a scope uses when nothing says otherwise. The collection has
#: measured numbers for this one, and it is the smallest of the candidates.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#: bge's documented instruction for the search side of a pair, and for nothing
#: else: the same words in front of a stored unit make the index worse.
_QUERY_INSTRUCTIONS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """What a model is, in the terms an index has to agree on.

    Two indexes built by different models, or by different dimensions of the
    same one, hold different spaces, and mixing them answers nonsense. The
    fingerprint is stored beside every vector so a mismatch can be detected
    rather than averaged over.
    """

    kind: str
    name: str
    dimension: int

    def __str__(self) -> str:
        return f"{self.name} ({self.dimension}d)"


class ModelError(RuntimeError):
    """Raised when a model cannot be loaded or does not do what it claims."""


@runtime_checkable
class Embedder(Protocol):
    """Turns text into unit vectors, one way for a unit and one for a query."""

    @property
    def identity(self) -> ModelIdentity: ...

    def warm(self) -> None:
        """Load now, so a later read finds the model resident."""

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


@runtime_checkable
class Reranker(Protocol):
    """Scores candidate units against a query, for ordering only."""

    @property
    def identity(self) -> ModelIdentity: ...

    def warm(self) -> None:
        """Load now, so a later read finds the model resident."""

    def score(self, query: str, candidates: Sequence[str]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    """One local model: what it is, how wide, and what it costs to fetch."""

    name: str
    dimension: int
    note: str


#: The local embedders, all of them reachable through the same interface. The
#: first is the default; the second is multilingual and much larger; the third
#: is the smallest and fastest, useful as the low end to measure against.
EMBEDDING_MODELS: dict[str, _ModelSpec] = {
    "BAAI/bge-small-en-v1.5": _ModelSpec(
        "BAAI/bge-small-en-v1.5", 384, "the default; the collection has numbers for it"
    ),
    "jinaai/jina-embeddings-v2-base-de": _ModelSpec(
        "jinaai/jina-embeddings-v2-base-de", 768, "multilingual, much larger"
    ),
    "sentence-transformers/all-MiniLM-L6-v2": _ModelSpec(
        "sentence-transformers/all-MiniLM-L6-v2", 384, "the smallest and fastest"
    ),
}

#: The local cross-encoders. The first two are small enough to score candidates
#: on a blocking read; the last two are the expensive end, where the collection
#: measured 2.3 s per query for 50 candidates.
RERANKER_MODELS: dict[str, _ModelSpec] = {
    "jinaai/jina-reranker-v1-tiny-en": _ModelSpec(
        "jinaai/jina-reranker-v1-tiny-en", 0, "small; affordable on a blocking read"
    ),
    "Xenova/ms-marco-MiniLM-L-6-v2": _ModelSpec(
        "Xenova/ms-marco-MiniLM-L-6-v2", 0, "small; affordable on a blocking read"
    ),
    "BAAI/bge-reranker-base": _ModelSpec(
        "BAAI/bge-reranker-base", 0, "measured at 2.3 s per 50 candidates"
    ),
    "jinaai/jina-reranker-v2-base-multilingual": _ModelSpec(
        "jinaai/jina-reranker-v2-base-multilingual", 0, "multilingual, large"
    ),
}


def available_embedders() -> tuple[str, ...]:
    """Return the local embedders, default first."""

    return (
        DEFAULT_EMBEDDING_MODEL,
        *(name for name in EMBEDDING_MODELS if name != DEFAULT_EMBEDDING_MODEL),
    )


def available_rerankers() -> tuple[str, ...]:
    """Return the local cross-encoders, cheapest first."""

    return tuple(RERANKER_MODELS)


def model_cache_directory(shared: str | Path | None = None) -> Path:
    """Return where this package keeps a fetched model.

    It is this package's own cache by default, never a sibling server's: a server
    that depends on another one's state is not independent, and a model is 65 MB
    that two packages would otherwise each fetch. ``runtime.model_cache_root`` in
    the settings points several projects on one machine at a single copy instead,
    which is a choice about disk rather than about what a memory contains.
    """

    if shared:
        return Path(shared).expanduser()
    return Path(user_cache_path("memory-ultra-rag-mcp", appauthor=False)) / "models"


def _normalise(vector: Sequence[float]) -> tuple[float, ...]:
    """Return one vector at unit length, so a dot product is a cosine."""

    total = math.sqrt(sum(value * value for value in vector))
    if not total:
        return tuple(float(value) for value in vector)
    return tuple(float(value) / total for value in vector)


class LocalEmbedder:
    """A local sentence embedder, loaded on first use and kept resident.

    Loading is the expensive part, so it is done once per process and never on
    a caller's thread: the server warms this on the write side, and a read that
    finds it unloaded reports that the semantic side was unavailable rather than
    paying for a load in the middle of a lookup.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        cache: Path | None = None,
    ) -> None:
        spec = EMBEDDING_MODELS.get(model)
        if spec is None:
            raise ModelError(
                f"unknown embedding model {model!r}; choose one of "
                + ", ".join(available_embedders())
            )
        self._spec = spec
        self._cache = model_cache_directory(cache)
        self._lock = threading.Lock()
        self._runtime: object | None = None

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity("embedding", self._spec.name, self._spec.dimension)

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def warm(self) -> None:
        """Load the model now, so a later read finds it resident."""

        self._runtime_for_write()

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        return self._embed(list(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        instruction = _QUERY_INSTRUCTIONS.get(self._spec.name, "")
        return self._embed([f"{instruction}{text}"])[0]

    def _runtime_for_write(self) -> object:
        """Return the loaded runtime, loading it once under the lock."""

        if self._runtime is not None:
            return self._runtime
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            self._runtime = self._load()
            return self._runtime

    def _load(self) -> object:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:  # pragma: no cover - a broken install
            raise ModelError(
                "the embedding library is not installed; run uv sync --frozen"
            ) from error
        cache = self._cache
        try:
            cache.mkdir(parents=True, exist_ok=True)
            return TextEmbedding(
                model_name=self._spec.name, cache_dir=str(cache), threads=1
            )
        except Exception as error:
            raise ModelError(
                f"{self._spec.name} could not be loaded from {cache}: {error}. "
                "It is fetched once from the model host and then runs locally; "
                "no API is used"
            ) from error

    def _embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        runtime = self._runtime_for_write()
        try:
            raw = list(runtime.embed(texts))  # type: ignore[attr-defined]
        except Exception as error:
            raise ModelError(f"{self._spec.name} failed to embed: {error}") from error
        if not raw:
            raise ModelError(
                f"{self._spec.name} returned nothing for {len(texts)} texts"
            )
        width = len(raw[0])
        if width != self._spec.dimension:
            raise ModelError(
                f"{self._spec.name} returned {width}-dimensional vectors, its table "
                f"says {self._spec.dimension}"
            )
        return [_normalise(vector) for vector in raw]


class LocalReranker:
    """A local cross-encoder, for ordering candidates on a read.

    Off by default. It is the largest ordering gain measured anywhere in this
    collection, and a small model makes it affordable on a blocking read, but it
    was measured on long passages rather than one-line units, and nothing in
    this package may claim a gain it has not measured.
    """

    def __init__(self, model: str, cache: Path | None = None) -> None:
        spec = RERANKER_MODELS.get(model)
        if spec is None:
            raise ModelError(
                f"unknown reranker model {model!r}; choose one of "
                + ", ".join(available_rerankers())
            )
        self._spec = spec
        self._cache = model_cache_directory(cache)
        self._lock = threading.Lock()
        self._runtime: object | None = None

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity("reranking", self._spec.name, 0)

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def warm(self) -> None:
        self._runtime_for_read()

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        runtime = self._runtime_for_read()
        try:
            # The runtime takes the query once and the documents beside it, and
            # returns one score per document in the order they were given.
            return [
                float(value)
                for value in runtime.rerank(  # type: ignore[attr-defined]
                    query, list(candidates)
                )
            ]
        except Exception as error:
            raise ModelError(f"{self._spec.name} failed to score: {error}") from error

    def _runtime_for_read(self) -> object:
        if self._runtime is not None:
            return self._runtime
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as error:  # pragma: no cover - a broken install
                raise ModelError(
                    "the reranking library is not installed; run uv sync --frozen"
                ) from error
            cache = self._cache
            try:
                cache.mkdir(parents=True, exist_ok=True)
                self._runtime = TextCrossEncoder(
                    model_name=self._spec.name, cache_dir=str(cache)
                )
            except Exception as error:
                raise ModelError(
                    f"{self._spec.name} could not be loaded from {cache}: {error}"
                ) from error
            return self._runtime


def describe_environment() -> dict[str, object]:
    """Report what the local models are, for a diagnostic surface or a test."""

    cache = model_cache_directory()
    return {
        "cache": str(cache),
        "cached": sorted(entry.split("models--")[-1] for entry in os.listdir(cache))
        if cache.is_dir()
        else [],
        "embedders": {
            name: {"dimension": spec.dimension, "note": spec.note}
            for name, spec in EMBEDDING_MODELS.items()
        },
        "rerankers": {
            name: {"note": spec.note} for name, spec in RERANKER_MODELS.items()
        },
    }
