"""Data models for parent-child chunking."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParentChildChunk:
    """A chunk with parent-child relationship for section-aware retrieval.

    - Parent chunk: the full section content (used for augmentation context)
    - Child chunk: a smaller piece of the section (used for search indexing)

    Retrieval flow:
        1. Search on child chunks (indexed by embedding)
        2. Get top child hits
        3. Map back to parent sections via parent_id
        4. Dedup parents
        5. Feed parent content into augmentation
    """

    chunk_id: str               # e.g. "1.00309__thanh_phan_ho_so__c0"
    parent_id: str              # e.g. "1.00309__thanh_phan_ho_so"
    content: str                # chunk text (child: 150-300 tokens; parent: full section)
    parent_content: str         # full section text (always the parent's content)
    metadata: dict = field(default_factory=dict)
    is_parent: bool = False     # True if this chunk IS the parent record

    @property
    def section_type(self) -> str:
        return self.metadata.get("section_type", "unknown")

    @property
    def ma_thu_tuc(self) -> str:
        return self.metadata.get("ma_thu_tuc", "")
