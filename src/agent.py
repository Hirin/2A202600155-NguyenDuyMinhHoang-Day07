"""Deterministic RAG orchestrator with corrective retry + self-check.

Replaces the LangGraph ReAct loop with a structured pipeline:
    1. Parse query → metadata filters + section intent
    2. Hybrid search with filters
    3. Judge retrieval quality → optional retry
    4. Augment → XML prompt
    5. Generate → structured JSON
    6. Self-check → corrections
    7. Return RAGResponse
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_openai import ChatOpenAI

from .augmentation.augmentor import Augmentor
from .generation.schemas import RAGResponse, RAGStatus
from .generation.self_check import SelfChecker
from .query.query_parser import QueryParser
from .retrieval.store import EmbeddingStore


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
DEFAULT_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
RETRIEVAL_SCORE_THRESHOLD = 0.4
MIN_EVIDENCE_COUNT = 1
MAX_RETRIES = 1


class KnowledgeBaseAgent:
    """Deterministic RAG orchestrator for TTHC administrative procedures.

    Pipeline:
        query_parser → search_with_filter → judge → [retry] →
        augmentor → LLM → self_check → RAGResponse
    """

    def __init__(
        self,
        store: EmbeddingStore,
        model: str = DEFAULT_LLM_MODEL,
    ) -> None:
        self.store = store
        self._model_name = model
        
        base_url = os.getenv("OPENAI_BASE_URL", None)
        api_key = os.getenv("OPENAI_API_KEY", "dummy-local-key" if base_url else None)
        
        self._llm = ChatOpenAI(
            model=model, 
            temperature=0, 
            api_key=api_key,
            base_url=base_url
        )
        self._parser = QueryParser()
        self._augmentor = Augmentor()
        self._checker = SelfChecker()

    def answer(self, question: str) -> str:
        """Run the full RAG pipeline and return the answer string.

        Returns the answer text for backward compatibility.
        Use answer_structured() for full RAGResponse.
        """
        response = self.answer_structured(question)
        return response.answer

    def answer_structured(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline and return structured RAGResponse."""

        # 1. Parse query
        parsed = self._parser.parse(question)

        # 2. Retrieve evidence
        evidence = self._retrieve(parsed.clean_query, parsed.metadata_filter)

        # 3. Judge retrieval quality → retry if needed
        if self._should_retry(evidence) and parsed.query_variants:
            variant = parsed.query_variants[0]
            retry_evidence = self._retrieve(variant, parsed.metadata_filter)
            if self._avg_score(retry_evidence) > self._avg_score(evidence):
                evidence = retry_evidence

        # 4. Early exit if no evidence
        if not evidence:
            return RAGResponse.insufficient(
                "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            )

        # 5. Build prompt
        prompt = self._augmentor.build_prompt(question, evidence)

        # 6. Generate
        try:
            raw_response = self._generate(prompt)
            response = self._parse_llm_output(raw_response)
        except Exception as e:
            return RAGResponse(
                answer=f"Lỗi khi xử lý câu hỏi: {e}",
                status=RAGStatus.INSUFFICIENT,
            )

        # 7. Self-check (Tier 1 always, Tier 2 conditional)
        query_context = {
            "section_intent": parsed.section_intent,
            "avg_retrieval_score": self._avg_score(evidence),
            "citation_count": len(response.citations),
        }
        response, check_result = self._checker.check(response, evidence, query_context)

        return response

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #
    def _retrieve(
        self,
        query: str,
        metadata_filter: dict[str, str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Run hybrid search with optional metadata filters."""
        if metadata_filter:
            return self.store.search_with_filter(
                query, top_k=top_k, metadata_filter=metadata_filter
            )
        return self.store.search(query, top_k=top_k)

    def _should_retry(self, evidence: list[dict]) -> bool:
        """Judge if retrieval quality is sufficient."""
        if len(evidence) < MIN_EVIDENCE_COUNT:
            return True
        if self._avg_score(evidence) < RETRIEVAL_SCORE_THRESHOLD:
            return True
        return False

    @staticmethod
    def _avg_score(evidence: list[dict]) -> float:
        """Average retrieval score."""
        if not evidence:
            return 0.0
        return sum(e.get("score", 0) for e in evidence) / len(evidence)

    def _generate(self, prompt: str) -> str:
        """Call LLM and return raw text response."""
        response = self._llm.invoke(prompt)
        return response.content

    def _parse_llm_output(self, raw: str) -> RAGResponse:
        """Parse LLM JSON output into RAGResponse.

        Handles:
        - Clean JSON
        - JSON wrapped in ```json ... ```
        - Fallback: treat as plain text answer
        """
        text = raw.strip()

        # Strip markdown code fence if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            return RAGResponse.from_dict(data)
        except json.JSONDecodeError:
            # Fallback: treat raw text as answer
            return RAGResponse(
                answer=raw.strip(),
                status=RAGStatus.GROUNDED,
            )
