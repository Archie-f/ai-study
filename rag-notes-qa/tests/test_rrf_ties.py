import pytest

from verify.verify_rrf_ties import count_tied_pairs, tie_crosses_cut


def test_count_tied_pairs():
    scored_lists = [
        [('a', 10.0), ('b', 10.0), ('c', 7.0)],
        [('a', 5.0), ('b', 5.0), ('c', 5.0)],
        [('a', 3.0), ('b', 2.0)],
        []
    ]
    result = [count_tied_pairs(scored) for scored in scored_lists]

    assert result == [1, 2, 0, 0]


def test_tie_crosses_cut_tie_exists():
    assert tie_crosses_cut([("a", 3.0), ("b", 2.8), ("c", 2.8), ("d", 1.9)], 2)


def test_tie_crosses_cut_tie_does_not_exist():
    assert not tie_crosses_cut([("a", 3.0), ("b", 2.8), ("c", 2.8), ("d", 2.3)], 3)


def test_tie_crosses_no_item_after_cut():
    assert not tie_crosses_cut([("a", 3.0), ("b", 2.8), ("c", 2.3), ("d", 1.9)], 4)


def test_tie_crosses_cut_raises_value_error():
    cut_values = [0, -1]
    for cut_value in cut_values:
        with pytest.raises(ValueError):
            tie_crosses_cut([("a", 3.0), ("b", 2.8), ("c", 2.3), ("d", 1.9)], cut_value)


