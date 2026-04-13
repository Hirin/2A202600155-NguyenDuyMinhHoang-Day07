"""Mock embedding backend — deterministic, used by tests and default runs."""
from __future__ import annotations

import hashlib
import math


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents/passages."""
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


_mock_embed = MockEmbedder()
