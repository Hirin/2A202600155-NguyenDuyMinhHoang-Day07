"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Chấm tự động bằng heuristic để lab chạy local:
  - Faithfulness
  - Answer relevance
  - Context recall
  - Completeness
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_answer import rag_answer

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

VARIANT_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "label": "variant_dense_rerank",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "bị", "bởi", "cho", "các", "có", "của",
    "đã", "để", "from", "in", "is", "khi", "không", "là", "một", "này", "như",
    "of", "on", "or", "the", "theo", "to", "trong", "từ", "và", "với",
}


# =============================================================================
# HELPERS
# =============================================================================

def _tokens(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[0-9A-Za-zÀ-ỹ]+", text)
        if token and token.lower() not in STOPWORDS
    ]


def _coverage_ratio(reference: str, candidate: str) -> float:
    reference_tokens = set(_tokens(reference))
    candidate_tokens = set(_tokens(candidate))
    if not reference_tokens:
        return 1.0
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def _ratio_to_score(ratio: float) -> int:
    if ratio >= 0.85:
        return 5
    if ratio >= 0.65:
        return 4
    if ratio >= 0.45:
        return 3
    if ratio >= 0.2:
        return 2
    return 1


def _is_abstain(answer: str) -> bool:
    lowered = answer.lower()
    return "không đủ dữ liệu" in lowered or "không tìm thấy thông tin" in lowered


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_faithfulness(answer: str, chunks_used: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not answer:
        return {"score": 1, "notes": "Answer rỗng"}

    if _is_abstain(answer):
        return {"score": 5, "notes": "Abstain an toàn, không bịa thông tin"}

    context_text = " ".join(chunk.get("text", "") for chunk in chunks_used)
    ratio = _coverage_ratio(answer, context_text)
    return {
        "score": _ratio_to_score(ratio),
        "notes": f"Token support ratio={ratio:.2f}",
    }


def score_answer_relevance(query: str, answer: str) -> Dict[str, Any]:
    if not answer:
        return {"score": 1, "notes": "Answer rỗng"}

    ratio = _coverage_ratio(query, answer)
    if _is_abstain(answer):
        if any(keyword in query.lower() for keyword in ("err-403-auth", "mức phạt", "vip")):
            return {"score": 5, "notes": "Abstain phù hợp với câu hỏi thiếu context"}
        return {"score": 3, "notes": "Abstain nhưng chưa chắc câu hỏi thiếu dữ liệu"}

    return {
        "score": _ratio_to_score(ratio),
        "notes": f"Query-answer overlap={ratio:.2f}",
    }


def score_context_recall(chunks_used: List[Dict[str, Any]], expected_sources: List[str]) -> Dict[str, Any]:
    if not expected_sources:
        return {"score": 5, "recall": 1.0, "notes": "No expected sources; abstain case"}

    retrieved_sources = {
        chunk.get("metadata", {}).get("source", "")
        for chunk in chunks_used
    }
    found = 0
    missing = []
    for expected in expected_sources:
        expected_name = Path(expected).stem.lower()
        matched = any(expected_name in Path(source).stem.lower() for source in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources)
    return {
        "score": _ratio_to_score(recall),
        "recall": recall,
        "found": found,
        "missing": missing,
        "notes": f"Retrieved {found}/{len(expected_sources)} expected sources",
    }


def score_completeness(query: str, answer: str, expected_answer: str) -> Dict[str, Any]:
    if not expected_answer:
        return {"score": 5, "notes": "Không có expected_answer để so sánh"}

    if _is_abstain(answer):
        if "không tìm thấy thông tin" in expected_answer.lower() or "không đề cập" in expected_answer.lower():
            return {"score": 5, "notes": "Abstain khớp expected answer"}
        return {"score": 2, "notes": "Abstain nhưng expected answer có dữ liệu"}

    ratio = _coverage_ratio(expected_answer, answer)
    return {
        "score": _ratio_to_score(ratio),
        "notes": f"Expected-answer coverage={ratio:.2f}",
    }


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as file:
            test_questions = json.load(file)

    results = []
    label = config.get("label", "unnamed")

    print(f"\n{'=' * 70}")
    print(f"Chạy scorecard: {label}")
    print(f"Config: {config}")
    print("=" * 70)

    for question in test_questions:
        question_id = question["id"]
        query = question["question"]
        expected_answer = question.get("expected_answer", "")
        expected_sources = question.get("expected_sources", [])
        category = question.get("category", "")

        if verbose:
            print(f"\n[{question_id}] {query}")

        result = rag_answer(
            query=query,
            retrieval_mode=config.get("retrieval_mode", "dense"),
            top_k_search=config.get("top_k_search", 10),
            top_k_select=config.get("top_k_select", 3),
            use_rerank=config.get("use_rerank", False),
            verbose=False,
        )
        answer = result["answer"]
        chunks_used = result["chunks_used"]

        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        completeness = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            "sources": result["sources"],
            "faithfulness": faith["score"],
            "faithfulness_notes": faith["notes"],
            "relevance": relevance["score"],
            "relevance_notes": relevance["notes"],
            "context_recall": recall["score"],
            "context_recall_notes": recall["notes"],
            "completeness": completeness["score"],
            "completeness_notes": completeness["notes"],
            "config_label": label,
        }
        results.append(row)

        if verbose:
            print(f"  Answer: {answer}")
            print(
                f"  Faithful={faith['score']} | Relevant={relevance['score']} | "
                f"Recall={recall['score']} | Complete={completeness['score']}"
            )

    for metric in ("faithfulness", "relevance", "context_recall", "completeness"):
        scores = [row[metric] for row in results if row[metric] is not None]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"Average {metric}: {avg:.2f}")

    return results


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict[str, Any]],
    variant_results: List[Dict[str, Any]],
    output_csv: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    summary: Dict[str, Dict[str, float]] = {}

    print(f"\n{'=' * 70}")
    print("A/B Comparison: Baseline vs Variant")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':>10} {'Variant':>10} {'Delta':>8}")
    print("-" * 55)

    for metric in metrics:
        baseline_scores = [row[metric] for row in baseline_results if row[metric] is not None]
        variant_scores = [row[metric] for row in variant_results if row[metric] is not None]
        baseline_avg = sum(baseline_scores) / len(baseline_scores)
        variant_avg = sum(variant_scores) / len(variant_scores)
        delta = variant_avg - baseline_avg
        summary[metric] = {
            "baseline": baseline_avg,
            "variant": variant_avg,
            "delta": delta,
        }
        print(f"{metric:<20} {baseline_avg:>10.2f} {variant_avg:>10.2f} {delta:>+8.2f}")

    print(f"\n{'Câu':<6} {'Baseline':<12} {'Variant':<12} {'Better?':<10}")
    print("-" * 46)
    baseline_by_id = {row["id"]: row for row in baseline_results}
    for variant in variant_results:
        baseline = baseline_by_id[variant["id"]]
        baseline_total = sum(baseline[metric] for metric in metrics)
        variant_total = sum(variant[metric] for metric in metrics)
        better = "Variant" if variant_total > baseline_total else ("Baseline" if baseline_total > variant_total else "Tie")
        print(f"{variant['id']:<6} {baseline_total:<12} {variant_total:<12} {better:<10}")

    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=combined[0].keys())
            writer.writeheader()
            writer.writerows(combined)
        print(f"\nKết quả CSV đã lưu tại: {csv_path}")

    return summary


# =============================================================================
# REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict[str, Any]], label: str) -> str:
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    averages = {}
    for metric in metrics:
        scores = [row[metric] for row in results if row[metric] is not None]
        averages[metric] = sum(scores) / len(scores) if scores else 0.0

    markdown = [
        f"# Scorecard: {label}",
        f"Generated: {timestamp}",
        "",
        "## Summary",
        "",
        "| Metric | Average Score |",
        "|--------|---------------|",
    ]
    for metric in metrics:
        markdown.append(f"| {metric.replace('_', ' ').title()} | {averages[metric]:.2f}/5 |")

    markdown.extend([
        "",
        "## Per-Question Results",
        "",
        "| ID | Category | Faithful | Relevant | Recall | Complete | Answer |",
        "|----|----------|----------|----------|--------|----------|--------|",
    ])

    for row in results:
        short_answer = row["answer"].replace("\n", " ").strip()
        if len(short_answer) > 110:
            short_answer = short_answer[:107] + "..."
        markdown.append(
            f"| {row['id']} | {row['category']} | {row['faithfulness']} | {row['relevance']} | "
            f"{row['context_recall']} | {row['completeness']} | {short_answer} |"
        )

    return "\n".join(markdown) + "\n"


def generate_ab_markdown(summary: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "# A/B Comparison",
        "",
        "| Metric | Baseline | Variant | Delta |",
        "|--------|----------|---------|-------|",
    ]
    for metric, values in summary.items():
        lines.append(
            f"| {metric.replace('_', ' ').title()} | {values['baseline']:.2f} | "
            f"{values['variant']:.2f} | {values['delta']:+.2f} |"
        )
    return "\n".join(lines) + "\n"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)
    print(f"Loaded {len(questions)} test questions")

    baseline_results = run_scorecard(BASELINE_CONFIG, questions, verbose=True)
    variant_results = run_scorecard(VARIANT_CONFIG, questions, verbose=True)
    ab_summary = compare_ab(baseline_results, variant_results, output_csv="ab_comparison.csv")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "scorecard_baseline.md").write_text(
        generate_scorecard_summary(baseline_results, BASELINE_CONFIG["label"]),
        encoding="utf-8",
    )
    (RESULTS_DIR / "scorecard_variant.md").write_text(
        generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"]),
        encoding="utf-8",
    )
    (RESULTS_DIR / "ab_summary.md").write_text(generate_ab_markdown(ab_summary), encoding="utf-8")

    print("\nĐã ghi:")
    print(f"  - {RESULTS_DIR / 'scorecard_baseline.md'}")
    print(f"  - {RESULTS_DIR / 'scorecard_variant.md'}")
    print(f"  - {RESULTS_DIR / 'ab_summary.md'}")
