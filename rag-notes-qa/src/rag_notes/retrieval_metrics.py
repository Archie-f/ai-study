def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Return 1.0 if any of the top-k retrieved IDs is relevant, else 0.0.

    Args:
        retrieved_ids: chunk IDs in ranked order, best match first
        relevant_ids: the set of chunk IDs considered relevant for this query
        k: how many of the top results to check
    Returns:
        1.0 on a hit within the top k, 0.0 otherwise
    """
    return 1.0 if any(retrieved_id in relevant_ids for retrieved_id in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Return 1 / rank of the first relevant ID in retrieved_ids (1-indexed).

    Args:
        retrieved_ids: chunk IDs in ranked order, best match first
        relevant_ids: the set of chunk IDs considered relevant for this query
    Returns:
        1 / rank of the first hit, or 0.0 if nothing relevant was retrieved
    """
    for retrieved_id in retrieved_ids:
        if retrieved_id in relevant_ids:
            return 1 / (1 + retrieved_ids.index(retrieved_id))
    return 0.0


def mean_reciprocal_rank(scores: list[float]) -> float:
    """Average a list of per-query reciprocal_rank() scores into one MRR value.

    Args:
        scores: one reciprocal_rank() result per query
    Returns:
        the mean of scores, or 0.0 for an empty list
    """
    if len(scores) == 0:
        return 0.0
    return sum(scores) / len(scores)
