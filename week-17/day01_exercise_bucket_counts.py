def bucket_counts(values: list[int], edges: list[int]) -> list[int]:
    """Count how many values fall into each bucket.

    Args:
        values: response times in milliseconds
        edges: sorted upper limits of the buckets (inclusive)
    Returns:
        len(edges) + 1 counts; the last one is for values above the last edge
    """
    counts = [0] * (len(edges) + 1)
    for value in values:
        bucket = len(edges)  # default: the last bucket (above all edges)
        for i, edge in enumerate(edges):
            if value <= edge:
                bucket = i
                break  # first edge that fits wins
        counts[bucket] += 1
    return counts
