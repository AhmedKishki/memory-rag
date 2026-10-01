"""A deterministic stand-in for the local model, so the suite needs no model.

A real embedder would make every test fetch 65 MB and would make an assertion
about *which* unit a query finds depend on a model's opinion. This one maps words
to hand-chosen directions, so a cosine is arithmetic: "manuscript" and "draft"
point the same way, "tea" does not, and anything unrecognised points a third way.
That is enough to prove the two sides work, the floor holds, and the labels are
honest, which is what these tests are for.
"""

from __future__ import annotations

import re
import threading
import zlib
from collections.abc import Sequence

from memory_rag.models import ModelError, ModelIdentity

#: Words that are treated as meaning the same thing.
SYNONYMS = {
    "manuscript": "draft",
    "paper": "draft",
}

#: A word that means something else entirely.
UNRELATED = "tea"


class FakeEmbedder:
    """Vectors chosen by hand, and a record of what it was asked to embed."""

    def __init__(self, *, model: str = "fake/model", dimension: int = 3) -> None:
        self._model = model
        self._dimension = dimension
        self._loaded = False
        self.documents: list[str] = []
        self.queries: list[str] = []
        self.fails = False

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity("embedding", self._model, self._dimension)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def calls(self) -> int:
        return len(self.documents) + len(self.queries)

    def warm(self) -> None:
        self._loaded = True

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if self.fails:
            raise ModelError("the fake embedder was told to fail")
        self._loaded = True
        self.documents.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        if self.fails:
            raise ModelError("the fake embedder was told to fail")
        self._loaded = True
        self.queries.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> tuple[float, ...]:
        lowered = text.casefold()
        meaning = next(iter(SYNONYMS.values()))
        if meaning in lowered or any(word in lowered for word in SYNONYMS):
            return self._unit(0)
        if UNRELATED in lowered:
            return self._unit(1)
        return self._unit(2)

    def _unit(self, axis: int) -> tuple[float, ...]:
        vector = [0.0] * self._dimension
        vector[axis] = 1.0
        return tuple(vector)


class FakeNearEmbedder(FakeEmbedder):
    """Vectors built from word overlap, so two wordings of one sentence look alike.

    The plain fake maps text to one of three orthogonal units, which is enough to
    say "these match" and not enough to say "these are nearly the same" — and a
    near-duplicate threshold is about the second. This one hashes each word onto an
    axis, so texts sharing most of their words land close together and unrelated
    ones stay far apart, which is what that check has to tell apart.
    """

    def __init__(self, *, model: str = "fake/near", dimension: int = 64) -> None:
        super().__init__(model=model, dimension=dimension)

    def _vector(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimension
        for word in re.findall(r"\w+", str(text).casefold()):
            index = zlib.crc32(word.encode("utf-8")) % self._dimension
            vector[index] += 1.0
        total = sum(value * value for value in vector) ** 0.5
        if not total:
            vector[0] = 1.0
            return tuple(vector)
        return tuple(value / total for value in vector)


class FakeReranker:
    """Scores by how many of the query's words a candidate holds."""

    def __init__(self) -> None:
        self._loaded = False
        self.calls = 0

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity("reranking", "fake/reranker", 0)

    @property
    def loaded(self) -> bool:
        return self._loaded

    def warm(self) -> None:
        self._loaded = True

    def score(self, query: str, candidates: Sequence[str]) -> list[float]:
        self._loaded = True
        self.calls += 1
        words = {word for word in query.casefold().split() if len(word) > 2}
        return [
            float(len(words & {word for word in item.casefold().split()}))
            for item in candidates
        ]


class GatedEmbedder(FakeEmbedder):
    """An embedder that waits for a test to let it finish.

    A write queues its statement and returns, and the worker embeds it on its own
    thread, so a test that wants to look at the record *before* that happens needs
    the worker held. `gate` waits on `release`, and the number of batches it was
    asked for is `batches`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.batches = 0

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.batches += 1
        self.release.wait(5.0)
        return super().embed_documents(texts)
