"""Core chunking strategies: Fixed-size, Sentence, and Recursive."""
from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Tách câu bằng regex: cắt tại ". ", "! ", "? ", ".\n"
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))

        return chunks if chunks else [text]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\\n\\n", "\\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: văn bản đủ ngắn rồi
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Không còn separator nào → cắt cứng theo chunk_size
        if not remaining_separators:
            results = []
            for start in range(0, len(current_text), self.chunk_size):
                results.append(current_text[start : start + self.chunk_size])
            return results

        sep = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Thử tách với separator hiện tại
        if sep == "":
            # Separator rỗng = cắt từng ký tự → cắt cứng
            return self._split(current_text, next_separators)

        parts = current_text.split(sep)

        results: list[str] = []
        current_buffer = ""

        for part in parts:
            candidate = (current_buffer + sep + part) if current_buffer else part
            if len(candidate) <= self.chunk_size:
                current_buffer = candidate
            else:
                # Flush buffer trước
                if current_buffer:
                    results.append(current_buffer)
                # Nếu part quá dài → đệ quy với separator tiếp theo
                if len(part) > self.chunk_size:
                    results.extend(self._split(part, next_separators))
                    current_buffer = ""
                else:
                    current_buffer = part

        if current_buffer:
            results.append(current_buffer)

        return results if results else [current_text]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)
