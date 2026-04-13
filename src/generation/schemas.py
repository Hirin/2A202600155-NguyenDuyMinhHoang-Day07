"""RAG output schemas and status types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RAGStatus(str, Enum):
    """Status of a RAG response."""
    GROUNDED = "grounded"
    INSUFFICIENT = "insufficient"
    CONFLICT = "conflict"


@dataclass
class RAGFacts:
    """Structured facts extracted from evidence."""
    ten_thu_tuc: str = ""
    ma_thu_tuc: str = ""
    thoi_han: str = ""
    phi_le_phi: str = ""
    ho_so: list[str] = field(default_factory=list)
    co_quan: str = ""
    can_cu_phap_ly: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ten_thu_tuc": self.ten_thu_tuc,
            "ma_thu_tuc": self.ma_thu_tuc,
            "thoi_han": self.thoi_han,
            "phi_le_phi": self.phi_le_phi,
            "ho_so": self.ho_so,
            "co_quan": self.co_quan,
            "can_cu_phap_ly": self.can_cu_phap_ly,
        }


@dataclass
class RAGResponse:
    """Structured RAG response with citations and status."""
    answer: str = ""
    facts: RAGFacts = field(default_factory=RAGFacts)
    citations: list[str] = field(default_factory=list)
    status: RAGStatus = RAGStatus.GROUNDED
    suggested_procedures: list[dict[str, str]] = field(default_factory=list)
    debug_chunks: list[dict] = field(default_factory=list)
    alpha: float = 0.5

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "facts": self.facts.to_dict(),
            "citations": self.citations,
            "status": self.status.value,
            "suggested_procedures": self.suggested_procedures,
            "debug_chunks": self.debug_chunks,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RAGResponse:
        """Parse a dict (from LLM JSON output) into RAGResponse."""
        facts_data = data.get("facts", {})
        facts = RAGFacts(
            ten_thu_tuc=facts_data.get("ten_thu_tuc", ""),
            ma_thu_tuc=facts_data.get("ma_thu_tuc", ""),
            thoi_han=facts_data.get("thoi_han", ""),
            phi_le_phi=facts_data.get("phi_le_phi", ""),
            ho_so=facts_data.get("ho_so", []),
            co_quan=facts_data.get("co_quan", ""),
            can_cu_phap_ly=facts_data.get("can_cu_phap_ly", []),
        )

        status_str = data.get("status", "grounded")
        try:
            status = RAGStatus(status_str)
        except ValueError:
            status = RAGStatus.GROUNDED

        return cls(
            answer=data.get("answer", ""),
            facts=facts,
            citations=data.get("citations", []),
            status=status,
            suggested_procedures=data.get("suggested_procedures", []),
        )

    @classmethod
    def insufficient(cls, reason: str = "") -> RAGResponse:
        """Factory for insufficient data response."""
        return cls(
            answer=reason or "Không tìm thấy đủ thông tin để trả lời câu hỏi này.",
            status=RAGStatus.INSUFFICIENT,
        )
