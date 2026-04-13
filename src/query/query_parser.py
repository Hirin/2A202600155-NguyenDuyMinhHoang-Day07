"""Query parser for TTHC queries — extract metadata filters and section intent."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..parsing.section_map import SECTION_MAP


@dataclass
class ParsedQuery:
    """Result of parsing a user query."""

    original: str                                   # raw user query
    clean_query: str                                # query with extracted codes/numbers removed
    metadata_filter: dict[str, str] = field(default_factory=dict)
    section_intent: str | None = None               # target section type
    query_variants: list[str] = field(default_factory=list)  # rewritten queries for retry


# Agency name → agency_folder mapping
_AGENCY_MAP: dict[str, str] = {
    "bộ công thương": "BoCongThuong",
    "bộ công an": "BoCongAn",
    "bộ tư pháp": "BoTuPhap",
    "bộ y tế": "BoYTe",
    "bộ giáo dục": "BoGiaoDucVaDaoTao",
    "bộ giáo dục và đào tạo": "BoGiaoDucVaDaoTao",
    "bộ tài chính": "BoTaiChinh",
    "bộ ngoại giao": "BoNgoaiGiao",
    "bộ nội vụ": "BoNoiVu",
    "bộ quốc phòng": "BoQuocPhong",
    "bộ xây dựng": "BoXayDung",
    "bộ khoa học": "BoKhoaHocVaCongNghe",
    "bộ khoa học và công nghệ": "BoKhoaHocVaCongNghe",
    "bộ nông nghiệp": "BoNongNghiepVaMoiTruong",
    "bộ văn hóa": "BoVanHoaTheThaoVaDuLich",
    "bộ dân tộc": "BoDanTocVaTonGiao",
    "tòa án": "ToaAnNhanDan",
    "tòa án nhân dân": "ToaAnNhanDan",
    "thanh tra chính phủ": "ThanhTraChinhPhu",
    "ngân hàng nhà nước": "NganHangNhaNuocVietNam",
    "văn phòng trung ương đảng": "VanPhongTrungUongDang",
    "ban tổ chức trung ương": "BanToChucTrungUong",
}

# Intent keywords → canonical section type
# Longer phrases are weighted higher to avoid false matches from short generic words.
_INTENT_KEYWORDS: dict[str, list[str]] = {
    "phi_le_phi": ["phí", "lệ phí", "chi phí", "miễn phí", "giá", "tốn bao nhiêu"],
    "thoi_han_giai_quyet": ["thời hạn", "bao lâu", "thời gian giải quyết", "mất bao lâu"],
    "thanh_phan_ho_so": ["hồ sơ", "giấy tờ", "thành phần", "tờ khai", "nộp gì", "cần nộp"],
    "can_cu_phap_ly": ["căn cứ", "pháp lý", "nghị định", "thông tư", "căn cứ pháp lý", "luật nào", "văn bản pháp luật"],
    "trinh_tu_thuc_hien": ["trình tự", "các bước", "quy trình", "cách làm", "trình tự thực hiện"],
    "cach_thuc_thuc_hien": ["cách thức", "nộp online", "trực tuyến", "bưu chính", "nộp trực tiếp"],
    "co_quan_thuc_hien": ["cơ quan nào", "cơ quan", "đơn vị nào", "đơn vị", "ai thực hiện", "nơi nộp", "nộp ở đâu"],
    "doi_tuong_thuc_hien": ["đối tượng", "ai được", "ai có thể"],
    "yeu_cau_dieu_kien": ["yêu cầu", "điều kiện"],
    "ket_qua_thuc_hien": ["kết quả", "được cấp", "giấy phép"],
}

# Regex patterns
_MA_THU_TUC_RE = re.compile(r"\b(\d+\.\d{3,6})\b")
_DECREE_RE = re.compile(r"(\d+/\d{4}/[A-ZĐa-zđ]+-[A-ZĐa-zđ]+)")


class QueryParser:
    """Parse user queries into structured metadata filters and section intents.

    Examples:
        "thủ tục 1.00309 thời hạn bao lâu"
        → filter: {ma_thu_tuc: "1.00309"}, intent: "thoi_han_giai_quyet"

        "thủ tục của Bộ Công Thương cho doanh nghiệp"
        → filter: {agency_folder: "BoCongThuong"}, intent: None

        "phí lệ phí đăng ký kinh doanh"
        → filter: {}, intent: "phi_le_phi"
    """

    def parse(self, query: str) -> ParsedQuery:
        """Parse a query into filters and intent."""
        original = query.strip()
        lower = original.lower()
        clean = original

        metadata_filter: dict[str, str] = {}

        # 1. Extract ma_thu_tuc
        ma_match = _MA_THU_TUC_RE.search(original)
        if ma_match:
            raw_code = ma_match.group(1)
            parts = raw_code.split('.')
            # Normalize trailing zeros (match DB float truncation)
            metadata_filter["ma_thu_tuc"] = f"{parts[0]}.{parts[1].rstrip('0')}"
            clean = clean.replace(ma_match.group(0), "").strip()
        else:
            # Fallback: User forgets the dot (e.g., 2000460 -> 2.00046)
            no_dot_match = re.search(r"\b([123])(\d{4,6})\b", original)
            if no_dot_match:
                prefix = no_dot_match.group(1)
                suffix = no_dot_match.group(2).rstrip('0')
                metadata_filter["ma_thu_tuc"] = f"{prefix}.{suffix}"
                clean = clean.replace(no_dot_match.group(0), "").strip()

        # 2. Extract agency
        for agency_name, folder in _AGENCY_MAP.items():
            if agency_name in lower:
                metadata_filter["agency_folder"] = folder
                break

        # 3. Detect section intent
        section_intent = self._detect_intent(lower)

        # 4. Generate query variants for retry
        variants = self._generate_variants(clean, section_intent)

        return ParsedQuery(
            original=original,
            clean_query=clean,
            metadata_filter=metadata_filter,
            section_intent=section_intent,
            query_variants=variants,
        )

    def _detect_intent(self, lower_query: str) -> str | None:
        """Detect the most likely section intent from keywords.

        Uses weighted scoring: longer keyword matches score higher,
        preventing short generic words from overriding specific phrases.
        """
        best_intent: str | None = None
        best_score = 0.0

        for intent, keywords in _INTENT_KEYWORDS.items():
            score = 0.0
            for kw in keywords:
                if kw in lower_query:
                    # Weight by keyword length — longer = more specific = higher weight
                    score += len(kw)
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent

    def _generate_variants(self, clean_query: str, intent: str | None) -> list[str]:
        """Generate query rewrites for corrective retry."""
        variants: list[str] = []

        # Variant 1: just the clean query (no codes)
        if clean_query != clean_query.strip():
            variants.append(clean_query.strip())

        # Variant 2: add intent context
        if intent:
            from ..parsing.section_map import SECTION_DISPLAY_NAMES
            display = SECTION_DISPLAY_NAMES.get(intent, "")
            if display:
                variants.append(f"{clean_query} {display}")

        # Variant 3: simplified version
        words = clean_query.split()
        if len(words) > 5:
            variants.append(" ".join(words[:5]))

        return variants
