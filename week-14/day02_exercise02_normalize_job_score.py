def normalize_job_score(score: float, scale: int) -> float | None:
    """Normalize given score from a given scale to the 0.0–1.0 range.

    Args:
        score: score value to normalize
        scale: scale to normalize
    Returns:
        Normalized score in 0.0-1.0 range
    Raises:
        ValueError: if the score is zero
    """
    if scale == 0:
        raise ValueError("Scale cannot be zero")
    elif score is None or scale is None:
        return None
    else:
        return score / scale
