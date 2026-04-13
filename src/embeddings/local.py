"""Sentence Transformers local embedding backend."""
from __future__ import annotations


LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents/passages."""
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]
