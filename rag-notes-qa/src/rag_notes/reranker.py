from rag_notes.models import Chunk


def pick_top_by_score(
        candidates: list[tuple[str, float, Chunk]],
        scores: list[float],
        top_k: int
) -> list[tuple[str, float, Chunk]]:
    """Re-score candidates and keep the best top_k.

    Args:
        candidates: (chunk_id, old_score, chunk) tuples as returned by search()
        scores: new scores; scores[i] belongs to candidates[i]
        top_k: how many results to keep
    Returns:
        up to top_k (chunk_id, new_score, chunk) tuples, highest new score first;
        candidates with equal scores keep their original order
    Raises:
        ValueError: if candidates and scores have different lengths
    """
    if len(candidates) != len(scores):
        raise ValueError(f"Number of candidates ({len(candidates)}) and number of scores ({len(scores)}) do not match.")

    new_candidates_list = [(chunk_id, score, chunk) for (chunk_id, _, chunk), score in zip(candidates, scores)]
    sorted_list = sorted(new_candidates_list, key=lambda x: x[1], reverse=True)

    return sorted_list[:top_k]


def drop_below_threshold(
        results: list[tuple[str, float, Chunk]],
        min_score: float
) -> list[tuple[str, float, Chunk]]:
    """Drop results whose score is below min_score.

    Args:
        results: (chunk_id, score, chunk) tuples, e.g. from rerank()
        min_score: lowest accepted score; a result with exactly this score is kept
    Returns:
        the kept results in their original order; empty if none qualifies
    """
    return [result for result in results if result[1] >= min_score]
