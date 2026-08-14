from collections import defaultdict


critic_score_ranking = ["Starfall Odyssey", "Pixel Knights", "Rustlands", "Echo Drift", "Fenmoor"]
similarity_ranking = ["Echo Drift", "Rustlands", "Fenmoor", "Starfall Odyssey", "Pixel Knights"]


def compute_rrf_scores(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """Compute each item's combined RRF score across multiple ranked lists.

    Args:
        rankings: ranker name -> ordered list of item ids, best match first
        k: RRF damping constant
    Returns:
        dict mapping item id -> summed RRF score across all rankers it appears in
    """
    result = defaultdict(float)
    for ranking, game_titles in rankings.items():
        for index, game_title in enumerate(game_titles):
            rank = index+1
            score = 1 / (k + rank)
            result[game_title] += score

    return result


if __name__ == "__main__":
    rankings_dict: dict[str, list[str]] = {"critic_score_ranking": critic_score_ranking,
                                           "similarity_ranking": similarity_ranking}
    scores = compute_rrf_scores(rankings_dict)
    for title, score in scores.items():
        print(f"- {title}: {score:.3f}")