"""Embedding backends: Mock, Local, OpenAI, LMStudio, LlamaCpp."""
from __future__ import annotations

import os

from .mock import MockEmbedder, _mock_embed
from .openai_embed import OpenAIEmbedder
from .local import LocalEmbedder
from .lmstudio import LMStudioEmbedder

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LMSTUDIO_EMBEDDING_MODEL = os.getenv("LMSTUDIO_EMBEDDING_MODEL", "vietlegal-harrier-0.6b")
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"

__all__ = [
    "MockEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "LMStudioEmbedder",
    "_mock_embed",
    "LOCAL_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "LMSTUDIO_EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER_ENV",
]
