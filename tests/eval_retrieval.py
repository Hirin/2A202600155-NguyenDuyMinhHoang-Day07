"""Retrieval evaluation metrics.

Measures:
    - doc_hit@K: correct document appears in top-K results
    - section_hit@K: correct section appears in top-K results
    - filter_precision: when filter applied, all results match filter
    - duplicate_ratio: near-duplicate chunks in top-K

Usage:
    python -m tests.eval_retrieval [--subset N] [--top-k K]
"""
from __future__ import annotations

import json
from pathlib import Path


def doc_hit_at_k(results: list[dict], expected_ma: str, k: int = 3) -> bool:
    """Check if the expected document appears in top-K results."""
    if not expected_ma:
        return True  # no expected doc → vacuously true
    for r in results[:k]:
        meta = r.get("metadata", {})
        if meta.get("ma_thu_tuc") == expected_ma:
            return True
        if meta.get("doc_id", "").startswith(expected_ma.replace(".", "")):
            return True
    return False


def section_hit_at_k(results: list[dict], expected_section: str, k: int = 3) -> bool:
    """Check if the expected section type appears in top-K results."""
    if not expected_section:
        return True
    for r in results[:k]:
        meta = r.get("metadata", {})
        if meta.get("section_type") == expected_section:
            return True
    return False


def filter_precision(results: list[dict], metadata_filter: dict) -> float:
    """Precision of metadata filter: what fraction of results match the filter."""
    if not metadata_filter or not results:
        return 1.0
    matching = 0
    for r in results:
        meta = r.get("metadata", {})
        if all(meta.get(k) == v for k, v in metadata_filter.items()):
            matching += 1
    return matching / len(results)


def duplicate_ratio(results: list[dict], k: int = 5) -> float:
    """Fraction of near-duplicate chunks in top-K (by content prefix)."""
    if len(results) <= 1:
        return 0.0
    top = results[:k]
    prefixes = [r.get("content", "")[:150] for r in top]
    unique = len(set(prefixes))
    return 1.0 - (unique / len(prefixes))


def evaluate_retrieval(
    store,
    queries: list[dict],
    parser=None,
    top_k: int = 3,
) -> dict:
    """Run retrieval evaluation on benchmark queries.

    Args:
        store: EmbeddingStore instance.
        queries: List of benchmark query dicts (from benchmark_queries.json).
        parser: Optional QueryParser for metadata filter extraction.
        top_k: Number of results to evaluate.

    Returns:
        Dict with aggregate metrics + per-query results.
    """
    results_per_query = []
    doc_hits = 0
    section_hits = 0
    total_dup_ratio = 0.0
    total_filter_prec = 0.0
    n_with_filter = 0

    for q in queries:
        query_text = q["query"]
        expected_ma = q.get("expected_ma_thu_tuc")
        expected_sec = q.get("expected_section")

        # Extract metadata filter if parser available
        meta_filter = {}
        if parser:
            parsed = parser.parse(query_text)
            meta_filter = parsed.metadata_filter

        # Search
        if meta_filter:
            search_results = store.search_with_filter(
                query_text, top_k=top_k, metadata_filter=meta_filter
            )
            n_with_filter += 1
            total_filter_prec += filter_precision(search_results, meta_filter)
        else:
            search_results = store.search(query_text, top_k=top_k)

        # Metrics
        dh = doc_hit_at_k(search_results, expected_ma, k=top_k)
        sh = section_hit_at_k(search_results, expected_sec, k=top_k)
        dr = duplicate_ratio(search_results, k=top_k)

        doc_hits += int(dh)
        section_hits += int(sh)
        total_dup_ratio += dr

        results_per_query.append({
            "id": q["id"],
            "category": q["category"],
            "doc_hit": dh,
            "section_hit": sh,
            "duplicate_ratio": dr,
            "num_results": len(search_results),
        })

    n = len(queries) or 1
    return {
        "doc_hit_at_k": doc_hits / n,
        "section_hit_at_k": section_hits / n,
        "avg_duplicate_ratio": total_dup_ratio / n,
        "filter_precision": total_filter_prec / n_with_filter if n_with_filter else 1.0,
        "total_queries": len(queries),
        "top_k": top_k,
        "per_query": results_per_query,
    }


def main():
    """CLI entrypoint for retrieval evaluation."""
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate retrieval quality")
    ap.add_argument("--subset", type=int, default=None, help="Only use first N queries")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    # Load benchmark
    bench_path = Path(__file__).parent / "benchmark_queries.json"
    queries = json.loads(bench_path.read_text(encoding="utf-8"))
    if args.subset:
        queries = queries[:args.subset]

    print(f"[eval_retrieval] {len(queries)} queries, top_k={args.top_k}")
    print("[eval_retrieval] ⚠️  Requires indexed store. Run main.py first.")


if __name__ == "__main__":
    main()
