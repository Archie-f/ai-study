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
