"""TTHC Document Parser — structured parser for Vietnamese administrative procedures.

Parses markdown TTHC files (with JSON metadata block) into structured
TTHCDocument objects with normalized sections.
"""
from __future__ import annotations

import json
import re
import csv
from dataclasses import dataclass, field
from pathlib import Path

from .section_map import SECTION_MAP


# ================================================================== #
# Data models
# ================================================================== #
@dataclass
class TTHCSection:
    """A single section within a TTHC document."""

    section_type: str       # canonical key: "thanh_phan_ho_so", "phi_le_phi", etc.
    heading: str            # original heading text as it appeared in the document
    content: str            # full section text (may be long)

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class TTHCDocument:
    """Structured representation of a Vietnamese administrative procedure."""

    doc_id: str                             # e.g. "1.00309"
    ma_thu_tuc: str = ""
    ten_thu_tuc: str = ""
    quyet_dinh: str = ""
    linh_vuc: str = ""
    cap_thuc_hien: str = ""
    doi_tuong_thuc_hien: list[str] = field(default_factory=list)
    co_quan_thuc_hien: str = ""
    agency_folder: str = ""
    source_path: str = ""
    source_url: str = ""
    sections: list[TTHCSection] = field(default_factory=list)
    raw_metadata: dict = field(default_factory=dict)
    preamble: str = ""                      # text before first section heading

    def get_section(self, section_type: str) -> TTHCSection | None:
        """Get a section by its canonical type key."""
        for s in self.sections:
            if s.section_type == section_type:
                return s
        return None

    def get_all_section_types(self) -> list[str]:
        """Return list of canonical section types present in this document."""
        return [s.section_type for s in self.sections]

    @property
    def flat_metadata(self) -> dict:
        """Metadata dict suitable for embedding into chunk metadata."""
        return {
            "doc_id": self.doc_id,
            "ma_thu_tuc": self.ma_thu_tuc,
            "ten_thu_tuc": self.ten_thu_tuc,
            "quyet_dinh": self.quyet_dinh,
            "linh_vuc": self.linh_vuc,
            "cap_thuc_hien": self.cap_thuc_hien,
            "doi_tuong_thuc_hien": ", ".join(self.doi_tuong_thuc_hien),
            "co_quan_thuc_hien": self.co_quan_thuc_hien,
            "agency_folder": self.agency_folder,
            "source_path": self.source_path,
            "source_url": self.source_url,
        }


# ================================================================== #
# Parser
# ================================================================== #
class TTHCParser:
    """Parse markdown TTHC files into structured TTHCDocument objects.

    Expected input format (produced by scripts/preprocess_tthc.py):

        ```json
        { "ma_thu_tuc": "1.00309", "agency_folder": "BoCongThuong", ... }
        ```

        # Procedure Title

        ## Section Heading 1
        content...

        ## Section Heading 2
        content...
    """

    # Regex for ## headings (optionally with trailing colon/whitespace)
    _HEADING_RE = re.compile(r"^##\s+(.+?)[\s:]*$", re.MULTILINE)

    def __init__(self, ids_dir: str | Path | None = None):
        """Initialize parser and optionally load mapping from TTHC_IDs directory."""
        self._id_mapping = {}
        if ids_dir:
            self._load_id_mapping(Path(ids_dir))

    def _load_id_mapping(self, ids_dir: Path):
        """Load ma_thu_tuc to internal ID mapping from CSV files."""
        for file in ids_dir.rglob("*.csv"):
            try:
                with open(file, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        code = row.get("PROCEDURE_CODE")
                        internal_id = row.get("ID")
                        if code and internal_id:
                            self._id_mapping[str(code)] = str(internal_id)
            except Exception as e:
                print(f"[TTHCParser] Could not load mapping from {file.name}: {e}")

    def parse_file(self, file_path: str | Path) -> TTHCDocument:
        """Parse a single TTHC markdown file."""
        p = Path(file_path)
        raw = p.read_text(encoding="utf-8")
        agency_folder = p.parent.name

        # 1. Extract JSON metadata block
        metadata, body = self._extract_metadata_block(raw)
        metadata["agency_folder"] = agency_folder

        # 2. Extract title from first # heading
        title = self._extract_title(body)

        # 3. Parse sections
        sections = self._parse_sections(body)

        # 4. Extract preamble (text between title and first section)
        preamble = self._extract_preamble(body)

        # 5. Build TTHCDocument
        doi_tuong = metadata.get("doi_tuong_thuc_hien", [])
        if isinstance(doi_tuong, str):
            doi_tuong = [x.strip() for x in doi_tuong.split(",")]
            
        ma_thu_tuc = metadata.get("ma_thu_tuc", p.stem)
        source_url = ""
        internal_id = self._id_mapping.get(ma_thu_tuc)
        if internal_id:
            source_url = f"https://dichvucong.gov.vn/p/home/dvc-tthc-thu-tuc-hanh-chinh-chi-tiet.html?ma_thu_tuc={internal_id}"

        return TTHCDocument(
            doc_id=p.stem,
            ma_thu_tuc=ma_thu_tuc,
            ten_thu_tuc=title,
            quyet_dinh=metadata.get("quyet_dinh", ""),
            linh_vuc=metadata.get("linh_vuc", ""),
            cap_thuc_hien=metadata.get("cap_thuc_hien", ""),
            doi_tuong_thuc_hien=doi_tuong,
            co_quan_thuc_hien=metadata.get("co_quan_thuc_hien", ""),
            agency_folder=agency_folder,
            source_path=str(p),
            source_url=source_url,
            sections=sections,
            raw_metadata=metadata,
            preamble=preamble,
        )

    def parse_directory(self, dir_path: str | Path) -> list[TTHCDocument]:
        """Parse all .md files in a directory (recursive)."""
        p = Path(dir_path)
        docs = []
        for md_file in sorted(p.rglob("*.md")):
            try:
                doc = self.parse_file(md_file)
                docs.append(doc)
            except Exception as e:
                print(f"[TTHCParser] ⚠️  Failed to parse {md_file.name}: {e}")
        return docs

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _extract_metadata_block(self, raw: str) -> tuple[dict, str]:
        """Extract ```json ... ``` block and return (metadata, remaining_body)."""
        metadata: dict = {}
        body = raw

        if "```json" in raw:
            try:
                js_start = raw.index("```json") + 7
                js_end = raw.index("```", js_start)
                metadata = json.loads(raw[js_start:js_end])
                body = raw[js_end + 3:].strip()
            except (ValueError, json.JSONDecodeError):
                pass

        return metadata, body

    def _extract_title(self, body: str) -> str:
        """Extract title from the first # heading (not ##)."""
        match = re.search(r"^#\s+(.+?)$", body, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _extract_preamble(self, body: str) -> str:
        """Extract text between the title and the first ## heading."""
        # Find end of title line
        title_match = re.search(r"^#\s+.+?$", body, re.MULTILINE)
        if not title_match:
            start = 0
        else:
            start = title_match.end()

        # Find first ## heading
        section_match = self._HEADING_RE.search(body, start)
        if not section_match:
            return body[start:].strip()

        return body[start:section_match.start()].strip()

    def _parse_sections(self, body: str) -> list[TTHCSection]:
        """Split body into sections based on ## headings."""
        sections: list[TTHCSection] = []
        headings = list(self._HEADING_RE.finditer(body))

        if not headings:
            return sections

        for i, match in enumerate(headings):
            heading_text = match.group(1).strip()
            section_type = self._normalize_heading(heading_text)

            # Section content: from end of heading to start of next heading
            content_start = match.end()
            content_end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
            content = body[content_start:content_end].strip()

            if not content:
                continue

            sections.append(TTHCSection(
                section_type=section_type,
                heading=heading_text,
                content=content,
            ))

        return sections

    def _normalize_heading(self, heading: str) -> str:
        """Normalize a heading string to a canonical section key.

        Strips whitespace, trailing colons, and matches against SECTION_MAP.
        Falls back to a slugified version if no match found.
        """
        cleaned = heading.strip().rstrip(":").strip().lower()

        # Direct match
        if cleaned in SECTION_MAP:
            return SECTION_MAP[cleaned]

        # Partial match (heading contains a known key as substring)
        for key, canonical in SECTION_MAP.items():
            if key in cleaned:
                return canonical

        # Fallback: slugify
        slug = re.sub(r"[^a-z0-9\u00C0-\u024F\u1EA0-\u1EF9]+", "_", cleaned)
        slug = slug.strip("_")
        return slug or "unknown"
