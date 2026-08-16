from rag_notes.hybrid_search import build_rank_map, rrf_merge

box_office_ranking: dict[str, int] = {
    "Avatar": 1,
    "The Dark Knight": 2,
    "Inception": 3,
    "The Matrix": 4,
    "The Avengers": 5
}

critic_ratings_ranking: dict[str, int] = {
    "Pulp Fiction": 1,
    "The Dark Knight": 2,
    "Inception": 3,
    "The Matrix": 4,
    "Interstellar": 5
}


def test_build_rank_map():
    movies: list[str] = [
        "Inception",
        "The Dark Knight",
        "Interstellar",
        "Pulp Fiction",
        "The Matrix"
    ]
    rank_map = build_rank_map(movies)
    for index, movie in enumerate(movies):
        assert rank_map[movie] == index+1


def test_build_rank_map_empty_list():
    movies: list[str] = []
    rank_map = build_rank_map(movies)
    assert rank_map == {}


def test_rrf_merge():
    all_rankings = [box_office_ranking, critic_ratings_ranking]
    merged_ranking = rrf_merge(all_rankings)

    all_titles = set()
    for ranking in all_rankings:
        all_titles.update(ranking.keys())
    assert len(merged_ranking) == len(all_titles)

    merged = dict(merged_ranking)
    for title, actual_score in merged.items():
        expected_score = 0.0
        for ranking in all_rankings:
            if title in ranking:
                rank = ranking[title]
                expected_score += 1 / (60 + rank)
        assert expected_score == actual_score

    assert all(merged_ranking[i][1] >= merged_ranking[i+1][1] for i in range(len(merged_ranking)-1))


def test_rrf_merge_empty_dict():
    empty_dict = dict()
    merged_rankings_with_empty = rrf_merge([box_office_ranking, empty_dict])
    all_rankings_without_empty = rrf_merge([box_office_ranking])
    assert merged_rankings_with_empty == all_rankings_without_empty
