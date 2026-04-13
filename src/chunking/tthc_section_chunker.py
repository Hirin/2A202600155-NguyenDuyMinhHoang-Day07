"""Section-aware chunker for TTHC documents with parent-child relationships.

Each TTHC section becomes a parent chunk. If a section is longer than
child_max_chars, it is split into overlapping child chunks for indexing.
Child chunks keep a reference to their parent for context retrieval.
"""
from __future__ import annotations

from ..parsing.tthc_parser import TTHCDocument, TTHCSection
from .models import ParentChildChunk


class TTHCSectionChunker:
    """Section-aware chunker producing parent-child chunk pairs.

    Args:
        child_max_chars: Maximum character length for a child chunk (default 800).
        child_overlap: Character overlap between consecutive child chunks (default 100).
        min_section_chars: Sections shorter than this are kept as single parent (default 50).
    """

    def __init__(
        self,
        child_max_chars: int = 800,
        child_overlap: int = 100,
        min_section_chars: int = 50,
    ) -> None:
        self.child_max_chars = child_max_chars
        self.child_overlap = child_overlap
        self.min_section_chars = min_section_chars

    def chunk(self, tthc_doc: TTHCDocument) -> list[ParentChildChunk]:
        """Chunk a TTHCDocument into parent-child chunks.

        For each section:
            1. Create ONE parent chunk (full section content)
            2. If section > child_max_chars, split into child chunks
            3. Each child keeps reference to parent_id
            4. All chunks carry full document metadata

        Returns:
            List of ParentChildChunk objects (parents + children interleaved).
        """
        all_chunks: list[ParentChildChunk] = []
        base_meta = tthc_doc.flat_metadata

        # Handle preamble (text before first section heading)
        if tthc_doc.preamble and len(tthc_doc.preamble) >= self.min_section_chars:
            preamble_id = f"{tthc_doc.doc_id}__preamble"
            meta = {**base_meta, "section_type": "preamble", "chunk_type": "parent"}
            all_chunks.append(ParentChildChunk(
                chunk_id=preamble_id,
                parent_id=preamble_id,
                content=tthc_doc.preamble,
                parent_content=tthc_doc.preamble,
                metadata=meta,
                is_parent=True,
            ))

        # Process each section
        for section in tthc_doc.sections:
            section_chunks = self._chunk_section(tthc_doc.doc_id, section, base_meta)
            all_chunks.extend(section_chunks)

        return all_chunks

    def _chunk_section(
        self,
        doc_id: str,
        section: TTHCSection,
        base_meta: dict,
    ) -> list[ParentChildChunk]:
        """Chunk a single section into parent + optional children."""
        parent_id = f"{doc_id}__{section.section_type}"
        parent_meta = {
            **base_meta,
            "section_type": section.section_type,
            "section_heading": section.heading,
            "chunk_type": "parent",
        }

        # Parent chunk (always created)
        parent = ParentChildChunk(
            chunk_id=parent_id,
            parent_id=parent_id,
            content=section.content,
            parent_content=section.content,
            metadata=parent_meta,
            is_parent=True,
        )

        # If section is short enough, return only parent (no children needed)
        if len(section.content) <= self.child_max_chars:
            return [parent]

        # Split into child chunks
        children = self._split_into_children(
            parent_id=parent_id,
            text=section.content,
            parent_content=section.content,
            base_meta={
                **base_meta,
                "section_type": section.section_type,
                "section_heading": section.heading,
                "chunk_type": "child",
            },
        )

        # Return parent + children
        return [parent] + children

    def _split_into_children(
        self,
        parent_id: str,
        text: str,
        parent_content: str,
        base_meta: dict,
    ) -> list[ParentChildChunk]:
        """Split text into overlapping child chunks with sub-section detection."""
        children: list[ParentChildChunk] = []
        step = self.child_max_chars - self.child_overlap

        if step <= 0:
            step = self.child_max_chars

        parent_section = base_meta.get("section_type", "")

        idx = 0
        for start in range(0, len(text), step):
            end = min(start + self.child_max_chars, len(text))
            child_text = text[start:end].strip()

            if not child_text:
                continue

            # Detect if child content belongs to a different sub-section
            detected = self._detect_subsection(child_text)
            child_meta = {**base_meta, "child_index": idx}
            if detected and detected != parent_section:
                child_meta["section_type"] = detected

            children.append(ParentChildChunk(
                chunk_id=f"{parent_id}__c{idx}",
                parent_id=parent_id,
                content=child_text,
                parent_content=parent_content,
                metadata=child_meta,
                is_parent=False,
            ))
            idx += 1

            if end >= len(text):
                break

        return children

    @staticmethod
    def _detect_subsection(text: str) -> str | None:
        """Detect the dominant sub-section type from chunk content.

        Scans the first 300 chars for known section keywords (from SECTION_MAP).
        Also detects legal reference tables (decree numbers, 'Số ký hiệu') as
        can_cu_phap_ly even without explicit headings.

        Returns the canonical section key if a strong match is found,
        giving priority to keywords appearing earliest in the text.
        """
        import re as _re
        from ..parsing.section_map import SECTION_MAP

        lower = text[:300].lower()
        best_key: str | None = None
        best_pos = len(lower) + 1

        for heading, canonical in SECTION_MAP.items():
            pos = lower.find(heading)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_key = canonical

        # Detect legal reference table patterns (e.g. "42/2024/QH15", "Số ký hiệu")
        # Scan full text since legal refs often appear in the latter half of merged chunks
        if best_key is None:
            full_lower = text.lower()
            if "số ký hiệu" in full_lower or "ngày ban hành" in full_lower or "cơ quan ban hành" in full_lower:
                best_key = "can_cu_phap_ly"
            elif _re.search(r"\d+/\d{4}/[a-zđ]+-[a-zđ]+", full_lower):
                best_key = "can_cu_phap_ly"

        return best_key

