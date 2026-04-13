"""Augmentor — build structured XML prompts from retrieved evidence.

Handles:
- Context deduplication (remove near-duplicate chunks)
- Document reordering (strongest first, second-strongest last)
- Metadata tag formatting per evidence block
- Token budget enforcement
- XML prompt structure
"""
from __future__ import annotations

import json


class Augmentor:
    """Build structured prompts from retrieved evidence for grounded generation."""

    DEFAULT_TOKEN_BUDGET = 12000  # conservative for gpt-5.4-mini (400k context)

    SYSTEM_PROMPT = (
        "Bạn là trợ lý hành chính chuyên về Thủ tục Hành chính Việt Nam.\n"
        "CHỈ trả lời dựa trên context được cung cấp bên dưới.\n"
        "Mỗi claim quan trọng PHẢI có citation dạng [mã_thủ_tục|tên_section].\n"
        "Nếu có nhiều nguồn mâu thuẫn, ưu tiên nguồn mới hơn và ghi chú xung đột.\n"
        "Nếu không đủ dữ liệu để trả lời, set status='insufficient'.\n"
        "Trả lời bằng JSON theo output_schema."
    )

    OUTPUT_SCHEMA = json.dumps({
        "answer": "câu trả lời tự nhiên bằng tiếng Việt",
        "facts": {
            "ten_thu_tuc": "...",
            "ma_thu_tuc": "...",
            "thoi_han": "...",
            "phi_le_phi": "...",
            "ho_so": ["..."],
            "co_quan": "...",
            "can_cu_phap_ly": ["..."],
        },
        "citations": ["[mã|section]"],
        "status": "grounded | insufficient | conflict",
    }, ensure_ascii=False, indent=2)

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.token_budget = token_budget

    def build_prompt(
        self,
        question: str,
        evidence: list[dict],
        system_prompt: str | None = None,
    ) -> str:
        """Build a structured XML prompt from question + evidence.

        Steps:
            1. Deduplicate evidence by content similarity
            2. Reorder: strongest first, second-strongest last
            3. Format each evidence block with metadata tag
            4. Build XML prompt
            5. Enforce token budget (truncate least-relevant if over)
        """
        system = system_prompt or self.SYSTEM_PROMPT

        # 1. Dedup
        deduped = self._deduplicate(evidence)

        # 2. Reorder (lost-in-the-middle mitigation)
        reordered = self._reorder(deduped)

        # 3. Format evidence blocks
        context_blocks = self._format_evidence(reordered)

        # 4. Enforce token budget
        context_text = self._enforce_budget(context_blocks)

        # 5. Build XML prompt
        return self._build_xml(system, context_text, question)

    def _deduplicate(self, evidence: list[dict], threshold: float = 0.9) -> list[dict]:
        """Remove near-duplicate chunks by content prefix overlap."""
        if not evidence:
            return []

        seen_prefixes: set[str] = set()
        unique: list[dict] = []

        for item in evidence:
            # Use first 150 chars as dedup key
            prefix = item.get("content", "")[:150].strip()
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            unique.append(item)

        return unique

    def _reorder(self, evidence: list[dict]) -> list[dict]:
        """Reorder: strongest first, second-strongest last.

        This mitigates the 'lost-in-the-middle' phenomenon where LLMs
        attend less to evidence in the middle of the context window.
        """
        if len(evidence) <= 2:
            return evidence

        # Sort by score descending
        scored = sorted(evidence, key=lambda x: x.get("score", 0), reverse=True)

        # Best → first, second-best → last, rest in middle
        reordered = [scored[0]]
        middle = scored[2:]
        reordered.extend(middle)
        reordered.append(scored[1])

        return reordered

    def _format_evidence(self, evidence: list[dict]) -> list[str]:
        """Format each evidence item with metadata tag."""
        blocks: list[str] = []

        for item in evidence:
            meta = item.get("metadata", {})
            ma = meta.get("ma_thu_tuc", "?")
            section = meta.get("section_type", "?")
            agency = meta.get("agency_folder", "?")
            content = item.get("content", "")

            tag = f"[{ma}|{section}|{agency}]"
            blocks.append(f"{tag} {content}")

        return blocks

    def _enforce_budget(self, blocks: list[str]) -> str:
        """Join blocks, truncating from the end if over token budget.

        Rough estimate: 1 token ≈ 4 chars for Vietnamese text.
        """
        char_budget = self.token_budget * 4
        result_parts: list[str] = []
        current_chars = 0

        for block in blocks:
            if current_chars + len(block) > char_budget:
                # Truncate this block to fit
                remaining = char_budget - current_chars
                if remaining > 100:  # only add if meaningful
                    result_parts.append(block[:remaining] + "...")
                break
            result_parts.append(block)
            current_chars += len(block)

        return "\n\n".join(result_parts)

    def _build_xml(self, system: str, context: str, question: str) -> str:
        """Assemble the final XML-structured prompt."""
        return f"""<system>
{system}
</system>

<context>
{context}
</context>

<question>{question}</question>

<output_schema>
{self.OUTPUT_SCHEMA}
</output_schema>"""
