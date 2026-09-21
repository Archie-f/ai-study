recommended_ids_positive = ['m5', 'm2', 'm9', 'm1', 'm7']
recommended_ids_negative = ['x1', 'x2']
watched_ids_positive = {'m9', 'm3'}
watched_ids_negative = {'m9'}



def movie_recommendation_hit(recommended_ids: list[str], watched_ids: set[str], k: int = 3) -> float:
    """Return 1.0 if any of the top-k recommended IDs is relevant, else 0.0.

    Args:
        recommended_ids: movie IDs in ranked order, best match first
        watched_ids: the set of movie IDs considered relevant for this query
        k: how many of the top results to check
    Returns:
        1.0 on a hit within the top k, 0.0 otherwise
    """
    return 1.0 if any(id in watched_ids for id in recommended_ids[:k]) else 0.0



if __name__ == '__main__':
    print(f"Happy path: {movie_recommendation_hit(recommended_ids_positive, watched_ids_positive)}")
    print(f"Negative scenario: {movie_recommendation_hit(recommended_ids_negative, watched_ids_negative)}")