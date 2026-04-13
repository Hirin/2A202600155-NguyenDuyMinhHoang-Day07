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
