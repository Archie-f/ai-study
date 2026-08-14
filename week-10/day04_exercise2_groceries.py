from rag_notes.hybrid_search import build_rank_map, rrf_merge


def merge_product_rankings(
        price_ranking: list[str],
        rating_ranking: list[str], k: int = 60,
) -> list[tuple[str, float]]:
    """Combine a price-based ranking and a rating-based ranking into one
    RRF-merged product ranking.

    Args:
        price_ranking: product ids, cheapest first
        rating_ranking: product ids, highest-rated first
        k: RRF damping constant
    Returns:
        list of (product_id, rrf_score) tuples, best combined match first
    """
    all_products = set()
    for ranking in (price_ranking, rating_ranking):
        all_products.update(set(ranking))

    all_rankings = [price_ranking, rating_ranking]
    mapped_rankings: list[dict[str, int]] = []
    for ranking in all_rankings:
        mapped_rankings.append(build_rank_map(ranking))

    merged = rrf_merge(mapped_rankings)
    for product_id, score in merged:
        print(f"{product_id}: {score:.3f}")

    return merged


price_ranking = ["oat-milk", "almond-milk", "soy-milk", "whole-milk"]
rating_ranking = ["whole-milk", "oat-milk", "soy-milk", "almond-milk"]


if __name__ == "__main__":
    merge_product_rankings(price_ranking, rating_ranking)