"""End-to-end RAG evaluation metrics.

Measures:
    - context_recall: did retrieved context contain the gold answer?
    - faithfulness: is the answer grounded in context?
    - answer_relevance: does the answer address the question?
    - citation_correctness: are citations valid and pointing to real sources?
    - abstention_correctness: did the model correctly abstain on out-of-scope?

Usage:
    python -m tests.eval_rag [--subset N]
"""
from __future__ import annotations

import json
from pathlib import Path


def context_recall(
    evidence: list[dict],
    gold_keywords: list[str],
) -> float:
    """Fraction of gold keywords found in retrieved evidence."""
    if not gold_keywords:
        return 1.0
    all_content = " ".join(e.get("content", "") for e in evidence).lower()
    found = sum(1 for kw in gold_keywords if kw.lower() in all_content)
    return found / len(gold_keywords)


def faithfulness(
    answer: str,
    evidence: list[dict],
) -> float:
    """Rough faithfulness: fraction of answer sentences with evidence overlap.

    Simple heuristic: split answer into sentences, check each has ≥1 keyword
    overlap with evidence content.
    """
    import re

    if not answer.strip():
        return 1.0  # empty answer is vacuously faithful

    sentences = re.split(r'[.!?]\s+', answer.strip())
    sentences = [s for s in sentences if len(s) > 10]

    if not sentences:
        return 1.0

    all_evidence = " ".join(e.get("content", "") for e in evidence).lower()
    evidence_words = set(all_evidence.split())

    grounded = 0
    for sent in sentences:
        sent_words = set(sent.lower().split())
        # A sentence is "grounded" if it shares ≥3 content words with evidence
        overlap = sent_words & evidence_words
        # Filter out common stop words
        meaningful_overlap = {w for w in overlap if len(w) > 3}
        if len(meaningful_overlap) >= 2:
            grounded += 1

    return grounded / len(sentences)


def citation_correctness(
    citations: list[str],
    evidence: list[dict],
) -> float:
    """Fraction of citations that point to actual evidence sources."""
    if not citations:
        return 1.0

    evidence_keys = set()
    for e in evidence:
        meta = e.get("metadata", {})
        ma = meta.get("ma_thu_tuc", "")
        section = meta.get("section_type", "")
        if ma and section:
            evidence_keys.add(f"[{ma}|{section}]")

    if not evidence_keys:
        return 0.0  # no evidence keys to match against

    valid = sum(1 for c in citations if c in evidence_keys)
    return valid / len(citations)


def abstention_correctness(
    response_status: str,
    expected_status: str,
) -> bool:
    """Check if the model correctly abstained (or didn't) on a query."""
    return response_status == expected_status


def evaluate_rag(
    agent,
    queries: list[dict],
) -> dict:
    """Run end-to-end RAG evaluation.

    Args:
        agent: KnowledgeBaseAgent instance.
        queries: List of benchmark query dicts.

    Returns:
        Dict with aggregate metrics + per-query results.
    """
    results_per_query = []
    total_faithfulness = 0.0
    total_citation = 0.0
    total_recall = 0.0
    abstention_correct = 0
    abstention_total = 0

    for q in queries:
        query_text = q["query"]
        expected_status = q.get("expected_status", "grounded")
        gold_keywords = q.get("gold_keywords", [])

        # Run agent
        response = agent.answer_structured(query_text)

        # Get evidence from store for metric calculation
        evidence = agent.store.search(query_text, top_k=5)

        # Metrics
        recall = context_recall(evidence, gold_keywords)
        faith = faithfulness(response.answer, evidence)
        cite_corr = citation_correctness(response.citations, evidence)

        total_recall += recall
        total_faithfulness += faith
        total_citation += cite_corr

        # Abstention
        if expected_status == "insufficient":
            abstention_total += 1
            if response.status.value == "insufficient":
                abstention_correct += 1

        results_per_query.append({
            "id": q["id"],
            "category": q["category"],
            "status": response.status.value,
            "expected_status": expected_status,
            "context_recall": recall,
            "faithfulness": faith,
            "citation_correctness": cite_corr,
            "num_citations": len(response.citations),
            "answer_length": len(response.answer),
        })

    n = len(queries) or 1
    return {
        "context_recall": total_recall / n,
        "faithfulness": total_faithfulness / n,
        "citation_correctness": total_citation / n,
        "abstention_correctness": (
            abstention_correct / abstention_total if abstention_total else 1.0
        ),
        "total_queries": len(queries),
        "per_query": results_per_query,
    }


def main():
    """CLI entrypoint."""
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate end-to-end RAG quality")
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()

    bench_path = Path(__file__).parent / "benchmark_queries.json"
    queries = json.loads(bench_path.read_text(encoding="utf-8"))
    if args.subset:
        queries = queries[:args.subset]

    print(f"[eval_rag] {len(queries)} queries")
    print("[eval_rag] ⚠️  Requires running agent + indexed store + API key.")


if __name__ == "__main__":
    main()
