"""Embedding protocol — all embedders must implement this interface.

Separates query embedding (may need instruction prefix) from
document embedding (raw passage, no prefix).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Standard interface for all embedding backends.

    Why two methods instead of one?
    - Some models (VietLegal-Harrier, Instructor, E5) require an instruction
      prefix for queries but NOT for passages.
    - Splitting into embed_query / embed_documents makes the asymmetry
      explicit and prevents bugs in the retrieval pipeline.
    """

    _backend_name: str

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. May apply model-specific instruction prefix."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passages/documents. No instruction prefix."""
        ...

def get_embedder_by_name() -> EmbedderProtocol:
    """Factory to instantiate the correct embedder based on EMBEDDING_PROVIDER in .env"""
    import os
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").lower()
    
    if provider == "openai":
        from .openai_embed import OpenAIEmbedder
        return OpenAIEmbedder()
    elif provider == "llamacpp":
        # Check if they have the module llamacpp.py (vietlegal harrier wrapper)
        from .llamacpp import LlamaCppEmbedder
        url = os.getenv("LLAMACPP_SERVER_URL", "http://localhost:8086")
        return LlamaCppEmbedder(server_url=url)
    elif provider == "lmstudio":
        from .lmstudio import LMStudioEmbedder
        url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        model = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "vietlegal-harrier-0.6b")
        return LMStudioEmbedder(base_url=url, model_name=model)
    elif provider == "local":
        from .local import LocalEmbedder
        return LocalEmbedder()
    else:
        from .mock import MockEmbedder
        return MockEmbedder()
