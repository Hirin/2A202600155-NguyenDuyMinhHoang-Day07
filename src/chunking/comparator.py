"""Chunking strategy comparator for A/B analysis."""
from __future__ import annotations

from .base import FixedSizeChunker, RecursiveChunker, SentenceChunker


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_size // 10),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        result = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            lengths = [len(c) for c in chunks]
            result[name] = {
                "count": len(chunks),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0.0,
                "chunks": chunks,
            }

        return result
