import pytest

from rag_notes.hybrid_search import rrf_merge
from verify.verify_candidate_options import weighted_rrf_merge, build_candidates


def test_equal_weights_match_rrf_merge():
    """Weights [1.0, 1.0] give exactly the same result as rrf_merge."""
    dense = {"a": 1, "b": 2, "c": 3}
    sparse = {"c": 1, "d": 2, "a": 3}

    # == is safe here: both functions add the same terms in the same order
    assert weighted_rrf_merge([dense, sparse], [1.0, 1.0]) == rrf_merge([dense, sparse])


def test_weight_changes_the_order():
    """With weights [2.0, 1.0] the worked example gives b, a, c."""
    result = weighted_rrf_merge([{"a": 1, "b": 2}, {"b": 1, "c": 2}], [2.0, 1.0])

    assert [chunk_id for chunk_id, _ in result] == ["b", "a", "c"]
    assert result[0][1] == pytest.approx(2 / 62 + 1 / 61)
    assert result[1][1] == pytest.approx(2 / 61)
    assert result[2][1] == pytest.approx(1 / 62)


def test_heavier_ranker_wins_against_equal_weights():
    """Rank 1 in the heavier list beats rank 1 in the lighter list; with equal weights the id decides."""
    dense = {"z": 1}
    sparse = {"a": 1}

    equal = weighted_rrf_merge([dense, sparse], [1.0, 1.0])
    heavy_dense = weighted_rrf_merge([dense, sparse], [2.0, 1.0])

    assert [chunk_id for chunk_id, _ in equal] == ["a", "z"]
    assert [chunk_id for chunk_id, _ in heavy_dense] == ["z", "a"]


def test_different_lengths_raise():
    """rank_maps and weights of different lengths raise ValueError."""
    with pytest.raises(ValueError):
        weighted_rrf_merge([{"a": 1}], [1.0, 1.0])


@pytest.mark.parametrize("weights", [[0, 1.0], [1.0, -1.0]])
def test_non_positive_weight_raises(weights):
    """A weight of 0 or less (in any position) raises ValueError."""
    with pytest.raises(ValueError):
        weighted_rrf_merge([{"a": 1}, {"b": 1}], weights)


def test_equal_scores_are_ordered_by_id():
    """Two chunks with the same score come out in id order."""
    result = weighted_rrf_merge([{"z": 1}, {"a": 1}], [1.0, 1.0])

    assert [chunk_id for chunk_id, _ in result] == ["a", "z"]


def test_empty_input_gives_empty_list():
    """No rankers and no weights give an empty list."""
    assert weighted_rrf_merge([], []) == []


def test_custom_k_is_used():
    """A non-default k changes the score: weight / (k + rank) with k=10."""
    result = weighted_rrf_merge([{"a": 1}], [2.0], k=10)

    assert result == [("a", pytest.approx(2 / 11))]


# build_candidates: lists used below (dense, sparse), k=60, default weights [1.0, 1.0]
DENSE = ["a", "b", "c"]
SPARSE = ["c", "d", "e"]


def test_build_candidates_depth_limits_what_each_ranker_contributes():
    """With depth 2 only a, b (dense) and c, d (sparse) are merged; a and c tie at rank 1, id decides."""
    assert build_candidates(DENSE, SPARSE, depth=2, cut=2) == ["a", "c"]


def test_build_candidates_depth_drops_ids_beyond_it_from_both_lists():
    """With depth 2 and a large cut, dense c (rank 3) and sparse e (rank 3) are gone: a, c, b, d remain."""
    assert build_candidates(DENSE, SPARSE, depth=2, cut=10) == ["a", "c", "b", "d"]


def test_build_candidates_dense_weight_changes_the_candidates():
    """With weights [2.0, 1.0] dense counts double, so b beats c."""
    assert build_candidates(DENSE, SPARSE, depth=2, cut=2, weights=[2.0, 1.0]) == ["a", "b"]


def test_build_candidates_chunk_in_both_lists_comes_first():
    """c is rank 3 in dense and rank 1 in sparse, so the two scores add up and c leads."""
    assert build_candidates(DENSE, SPARSE, depth=3, cut=10) == ["c", "a", "b", "d", "e"]


def test_build_candidates_cut_longer_than_the_merged_list_is_fine():
    """A cut larger than the number of distinct ids returns all of them without error."""
    assert len(build_candidates(DENSE, SPARSE, depth=3, cut=100)) == 5


def test_build_candidates_depth_larger_than_the_lists_is_fine():
    """A depth larger than the lists behaves like using the whole lists."""
    assert build_candidates(DENSE, SPARSE, depth=100, cut=10) == build_candidates(DENSE, SPARSE, depth=3, cut=10)


@pytest.mark.parametrize("depth", [0, -1])
def test_build_candidates_non_positive_depth_raises(depth):
    """A depth of 0 or less raises ValueError (a negative slice would silently drop items)."""
    with pytest.raises(ValueError):
        build_candidates(DENSE, SPARSE, depth=depth, cut=2)


@pytest.mark.parametrize("cut", [0, -1])
def test_build_candidates_non_positive_cut_raises(cut):
    """A cut of 0 or less raises ValueError (a negative slice would silently drop items)."""
    with pytest.raises(ValueError):
        build_candidates(DENSE, SPARSE, depth=2, cut=cut)


def test_build_candidates_wrong_number_of_weights_raises():
    """One weight for two rankers raises ValueError."""
    with pytest.raises(ValueError):
        build_candidates(DENSE, SPARSE, depth=2, cut=2, weights=[1.0])
