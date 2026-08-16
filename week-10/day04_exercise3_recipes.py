from rag_notes.hybrid_search import rrf_merge, build_rank_map


def find_best_recipes(
        keyword_ranking: list[str],
        popularity_ranking: list[str],
        n: int = 3,
) -> list[tuple[str, float]]:
    """Combine a keyword-match ranking and a popularity ranking into one
    RRF-merged top-n recipe list.

    Args:
        keyword_ranking: recipe ids, best keyword match first
        popularity_ranking: recipe ids, most saved first
        n: how many merged results to return
    Returns:
        top n (recipe_id, rrf_score) tuples, best combined match first
    """
    return rrf_merge([build_rank_map(keyword_ranking), build_rank_map(popularity_ranking)])[:n]

keyword_ranking = ["lemon-risotto", "garlic-pasta", "miso-soup", "lemon-tart"]
popularity_ranking = ["garlic-pasta", "lemon-tart", "lemon-risotto", "miso-soup"]


if __name__ == "__main__":
    merged_ranking = find_best_recipes(keyword_ranking, popularity_ranking)
    for recipe in merged_ranking:
        print(f"{recipe[0]}: {recipe[1]:.3f}")