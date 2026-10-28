import pytest

from verify.verify_first_stage_recall import first_rank, count_within


def test_match_in_the_middle_returns_its_position():
    """A single relevant id at position 3 gives 3."""
    rank = first_rank(['a', 'b', 'c', 'd', 'e'], {'f', 'c'})
    assert rank == 3


def test_several_relevant_ids_return_the_first_position():
    """With two relevant ids, the earlier one decides the rank."""
    rank = first_rank(['a', 'b', 'c', 'd'], {'b', 'c', 'd'})
    assert rank == 2


def test_no_match_returns_none():
    """No relevant id in the list gives None."""
    rank = first_rank(['a', 'b', 'c', 'd', 'e'], {'f', 'g'})
    assert rank is None


def test_empty_ranked_list_returns_none():
    """An empty ranked list gives None."""
    rank = first_rank([], {'a', 'b'})
    assert rank is None


def test_empty_relevant_set_returns_none():
    """An empty relevant set gives None."""
    rank = first_rank(['a', 'b', 'c', 'd'], set())
    assert rank is None


def test_match_at_first_position_returns_one():
    """A match at the top gives 1, so counting starts at 1, not 0."""
    rank = first_rank(['a', 'b', 'c', 'd'], {'a', 'b', 'c'})
    assert rank == 1


def test_count_within_counts_ranks_up_to_each_cutoff():
    """Ranks [1, 5, None, 12] with cutoffs [3, 10, 20] give {3: 1, 10: 2, 20: 3}."""
    counts = count_within([1, 5, None, 12], [3, 10, 20])
    assert counts == {3: 1, 10: 2, 20: 3}


def test_count_within_empty_ranks_gives_zero_per_cutoff():
    """No ranks at all gives 0 for every cutoff."""
    counts = count_within([], [5, 3])
    assert counts == {5: 0, 3: 0}


def test_count_within_all_none_gives_zero():
    """Questions that were never found are not counted."""
    counts = count_within([None, None], [5, 3])
    assert counts == {5: 0, 3: 0}

def test_count_within_rank_equal_to_cutoff_is_counted():
    """A rank exactly at the cutoff counts (<=, not <)."""
    counts = count_within([3, 5, 10, 12], [3, 10])
    assert counts == {3: 1, 10: 3}


@pytest.mark.parametrize("cutoff", [0, -1])
def test_count_within_non_positive_cutoff_raises(cutoff):
    """A cutoff of 0 or less raises ValueError."""
    with pytest.raises(ValueError):
        count_within([3, 5, 10, 12], [cutoff])


def test_count_within_bad_cutoff_after_a_good_one_still_raises():
    """The check applies to every cutoff, not only the first."""
    with pytest.raises(ValueError):
        count_within([3, 5, 10, 12], [10, 0])
