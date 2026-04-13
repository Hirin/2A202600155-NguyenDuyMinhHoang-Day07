"""Chunking strategies for text splitting."""
from .base import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    _dot,
    compute_similarity,
)
from .comparator import ChunkingStrategyComparator

__all__ = [
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "ChunkingStrategyComparator",
    "compute_similarity",
    "_dot",
]
