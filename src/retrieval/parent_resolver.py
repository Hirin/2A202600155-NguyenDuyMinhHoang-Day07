"""Parent resolver: map child chunks back to parent sections with deduplication."""
from __future__ import annotations


def resolve_parents(
    child_results: list[dict],
    all_chunks: list[dict] | None = None,
) -> list[dict]:
    """Given search results (child hits), resolve and deduplicate parent sections.

    For each child hit:
        1. Look up parent_id in metadata
        2. Find the parent chunk (from all_chunks or from child's parent_content)
        3. Deduplicate by parent_id
        4. Preserve the best child score for each parent

    Args:
        child_results: Search results, each with metadata containing 'parent_id'.
        all_chunks: Optional full chunk list to look up parent content.
                    If None, uses 'parent_content' from child metadata if available.

    Returns:
        Deduplicated list of parent-level results with aggregated scores.
    """
    if not child_results:
        return []

    # Build parent lookup if all_chunks provided
    parent_lookup: dict[str, dict] = {}
    if all_chunks:
        for chunk in all_chunks:
            meta = chunk.get("metadata", {})
            if meta.get("chunk_type") == "parent":
                pid = meta.get("parent_id") or chunk.get("id", "")
                parent_lookup[pid] = chunk

    # Group children by parent_id
    seen_parents: dict[str, dict] = {}
    parent_order: list[str] = []

    for result in child_results:
        meta = result.get("metadata", {})
        parent_id = meta.get("parent_id", "")

        if not parent_id:
            # No parent relationship — keep as-is
            fallback_id = f"_standalone_{len(seen_parents)}"
            seen_parents[fallback_id] = result
            parent_order.append(fallback_id)
            continue

        if parent_id in seen_parents:
            # Already seen this parent — update score if better
            existing = seen_parents[parent_id]
            if result.get("score", 0) > existing.get("score", 0):
                seen_parents[parent_id]["score"] = result["score"]
            # Track child count
            seen_parents[parent_id].setdefault("_child_hits", 0)
            seen_parents[parent_id]["_child_hits"] += 1
            continue

        # First time seeing this parent
        parent_result = _build_parent_result(result, parent_id, parent_lookup)
        parent_result["_child_hits"] = 1
        seen_parents[parent_id] = parent_result
        parent_order.append(parent_id)

    # Return in order of first appearance (best-scoring child determines order)
    return [seen_parents[pid] for pid in parent_order]


def _build_parent_result(
    child_result: dict,
    parent_id: str,
    parent_lookup: dict[str, dict],
) -> dict:
    """Build a parent-level result from a child hit."""
    # Try to get parent content from lookup
    if parent_id in parent_lookup:
        parent = parent_lookup[parent_id]
        return {
            "content": parent.get("content", child_result["content"]),
            "score": child_result.get("score", 0.0),
            "metadata": {
                **parent.get("metadata", {}),
                "resolved_from": "parent_lookup",
            },
        }

    # Fallback: use parent_content from child metadata if available
    child_meta = child_result.get("metadata", {})
    parent_content = child_meta.get("parent_content", "")

    if parent_content:
        return {
            "content": parent_content,
            "score": child_result.get("score", 0.0),
            "metadata": {
                **child_meta,
                "chunk_type": "parent",
                "resolved_from": "child_parent_content",
            },
        }

    # Last fallback: just return the child result with parent_id noted
    return {
        "content": child_result["content"],
        "score": child_result.get("score", 0.0),
        "metadata": {
            **child_meta,
            "resolved_from": "child_fallback",
        },
    }
