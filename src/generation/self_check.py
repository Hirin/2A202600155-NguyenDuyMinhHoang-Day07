"""Two-tier self-check: rule-based first, LLM-based second.

Tier 1 (always runs, zero cost):
    - Citation existence: do cited [ma|section] appear in evidence?
    - Facts populated: are required fields non-empty for grounded status?
    - Status logic: does status match content? (empty answer → insufficient)
    - Duplicate citations: remove repeated citations

Tier 2 (conditional, 1 LLM call):
    Only triggered when:
    - Tier 1 found issues
    - Query is about legal recency / conflict
    - Retrieval confidence was low
    - Answer has 3+ claims to verify
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .schemas import RAGResponse, RAGStatus


@dataclass
class RuleCheckResult:
    """Result of Tier 1 rule-based check."""

    issues: list[str] = field(default_factory=list)
    corrections_applied: list[str] = field(default_factory=list)
    needs_llm_check: bool = False

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0


class SelfChecker:
    """Two-tier answer validation: cheap rules first, LLM only when needed."""

    def check(
        self,
        response: RAGResponse,
        evidence: list[dict],
        query_context: dict | None = None,
    ) -> tuple[RAGResponse, RuleCheckResult]:
        """Run self-check and return corrected response + check result.

        Args:
            response: The RAG response to validate.
            evidence: Retrieved evidence that was used for generation.
            query_context: Optional context (retrieval scores, query category).

        Returns:
            (corrected_response, check_result)
        """
        result = self._tier1_rule_check(response, evidence)

        # Apply automatic corrections
        response = self._apply_corrections(response, result)

        return response, result

    def should_trigger_llm_check(
        self,
        rule_result: RuleCheckResult,
        query_context: dict | None = None,
    ) -> bool:
        """Determine if Tier 2 LLM check should run.

        Triggers when:
            - Tier 1 found issues
            - Query deals with legal recency (can_cu_phap_ly intent)
            - Retrieval confidence is low (avg_score < 0.6)
            - Answer has many citations (4+)
        """
        if rule_result.needs_llm_check:
            return True

        ctx = query_context or {}

        # Legal recency queries need extra verification
        if ctx.get("section_intent") in ("can_cu_phap_ly",):
            return True

        # Low retrieval confidence
        if ctx.get("avg_retrieval_score", 1.0) < 0.6:
            return True

        # Many claims = higher risk
        if ctx.get("citation_count", 0) >= 4:
            return True

        return False

    def _tier1_rule_check(
        self,
        response: RAGResponse,
        evidence: list[dict],
    ) -> RuleCheckResult:
        """Fast deterministic validation — zero LLM cost."""
        issues: list[str] = []

        # 1. Citation existence check
        evidence_keys = set()
        for e in evidence:
            meta = e.get("metadata", {})
            ma = meta.get("ma_thu_tuc", "")
            section = meta.get("section_type", "")
            if ma and section:
                evidence_keys.add(f"[{ma}|{section}]")

        phantom_citations = []
        for citation in response.citations:
            if evidence_keys and citation not in evidence_keys:
                phantom_citations.append(citation)
                issues.append(f"phantom_citation:{citation}")

        # 2. Facts completeness (only check if status is grounded)
        if response.status == RAGStatus.GROUNDED:
            if not response.facts.ma_thu_tuc and not response.facts.ten_thu_tuc:
                issues.append("missing_fact:ma_thu_tuc_and_ten_thu_tuc")

        # 3. Status logic consistency
        if not response.answer.strip() and response.status == RAGStatus.GROUNDED:
            issues.append("empty_answer_but_grounded")

        if response.answer.strip() and response.status == RAGStatus.INSUFFICIENT:
            # Has a real answer but marked insufficient — might be wrong status
            if len(response.citations) > 0:
                issues.append("has_citations_but_insufficient")

        # 4. Duplicate citations
        if len(response.citations) != len(set(response.citations)):
            issues.append("duplicate_citations")

        return RuleCheckResult(
            issues=issues,
            needs_llm_check=len(phantom_citations) > 0 or len(issues) > 2,
        )

    def _apply_corrections(
        self,
        response: RAGResponse,
        result: RuleCheckResult,
    ) -> RAGResponse:
        """Apply automatic corrections based on Tier 1 findings."""
        corrections: list[str] = []

        # Remove phantom citations
        phantom_set = {
            issue.split(":", 1)[1]
            for issue in result.issues
            if issue.startswith("phantom_citation:")
        }
        if phantom_set:
            original_count = len(response.citations)
            response.citations = [c for c in response.citations if c not in phantom_set]
            corrections.append(
                f"removed {original_count - len(response.citations)} phantom citations"
            )

        # Deduplicate citations
        if "duplicate_citations" in result.issues:
            seen: list[str] = []
            for c in response.citations:
                if c not in seen:
                    seen.append(c)
            response.citations = seen
            corrections.append("deduplicated citations")

        # Fix status inconsistencies
        if "empty_answer_but_grounded" in result.issues:
            response.status = RAGStatus.INSUFFICIENT
            corrections.append("status changed: grounded → insufficient (empty answer)")

        result.corrections_applied = corrections
        return response
