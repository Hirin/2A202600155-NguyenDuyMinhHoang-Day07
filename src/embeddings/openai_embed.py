"""OpenAI embeddings API backend.

Supports both single-query and batch embedding via the OpenAI API.
Batch embedding (embed_documents) sends up to 500 texts per API call
for significantly better throughput during ingestion.
"""
from __future__ import annotations

import os


OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
MAX_BATCH_SIZE = 500  # OpenAI accepts up to 2048, but 500 is safer for token limits


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
        """Embed a single search query."""
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(v) for v in response.data[0].embedding]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents in batches (much faster than one-by-one).

        Sends up to MAX_BATCH_SIZE texts per API call.  Results are returned
        in the same order as the input texts.
        """
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            embeddings.extend(
                [float(v) for v in record.embedding]
                for record in response.data
            )
        return embeddings
