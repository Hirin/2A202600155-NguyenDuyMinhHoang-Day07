"""OpenAI embeddings API backend."""
from __future__ import annotations

import os


OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str | None = None) -> None:
        from openai import OpenAI

        self.model_name = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
        self._backend_name = self.model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents/passages."""
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]
