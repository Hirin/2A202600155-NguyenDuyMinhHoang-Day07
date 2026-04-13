"""Fusion algorithms: Reciprocal Rank Fusion (RRF) and Maximal Marginal Relevance (MMR)."""
from __future__ import annotations

import math


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """Merge N ranked lists via RRF. Each item must have 'content' key.

    RRF score = sum(1 / (k + rank_i)) across all lists where item appears.
    Higher k → more uniform weighting across ranks.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            key = item["content"][:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in items:
                items[key] = item

    merged = sorted(
        items.values(),
        key=lambda x: scores[x["content"][:200]],
        reverse=True,
    )
    return merged


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def mmr_rerank(
    candidates: list[dict],
    query_embedding: list[float],
    top_k: int,
    lambda_param: float = 0.7,
    embedding_key: str = "embedding",
) -> list[dict]:
    """Maximal Marginal Relevance re-ranking for diversity.

    Balances relevance to query vs novelty compared to already-selected items.

    MMR = argmax [ λ * sim(q, d_i) - (1-λ) * max(sim(d_i, d_j)) ]
          for d_j in already selected

    Args:
        candidates: List of result dicts with 'embedding' key.
        query_embedding: The query vector.
        top_k: Number of results to select.
        lambda_param: 0-1 tradeoff. Higher = more relevant, lower = more diverse.
        embedding_key: Key in candidate dict containing the embedding vector.

    Returns:
        Re-ranked list of top_k candidates with MMR scores.
    """
    if not candidates or top_k <= 0:
        return []

    # Filter candidates that have embeddings
    with_emb = [c for c in candidates if embedding_key in c]
    without_emb = [c for c in candidates if embedding_key not in c]

    # If no embeddings available, return originals by score
    if not with_emb:
        return candidates[:top_k]

    selected: list[dict] = []
    remaining = list(with_emb)

    for _ in range(min(top_k, len(with_emb))):
        best_score = float("-inf")
        best_idx = 0

        for i, cand in enumerate(remaining):
            cand_emb = cand[embedding_key]

            # Relevance to query
            relevance = _cosine_sim(query_embedding, cand_emb)

            # Max similarity to already selected (diversity penalty)
            max_sim_selected = 0.0
            for sel in selected:
                sim = _cosine_sim(cand_emb, sel[embedding_key])
                max_sim_selected = max(max_sim_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = best_score
        selected.append(chosen)

    # Fill remaining slots with non-embedded candidates if needed
    if len(selected) < top_k:
        selected.extend(without_emb[: top_k - len(selected)])

    return selected
