import pytest

from verify.verify_tie_options import keep_tied_at_cut, prefer_vector_side

SCORED = [('a', 0.5), ('b', 0.4), ('c', 0.4), ('d', 0.4), ('e', 0.3)]


def test_keep_tied_at_cut_keeps_all_items_tied_with_the_last_kept():
    """A tie across the cut keeps every tied item instead of cutting inside the tie group."""
    assert keep_tied_at_cut(SCORED, 2) == [('a', 0.5), ('b', 0.4), ('c', 0.4), ('d', 0.4)]


def test_keep_tied_at_cut_without_tie_cuts_exactly():
    """When the first dropped score differs, exactly cut items are kept."""
    assert keep_tied_at_cut(SCORED, 1) == [('a', 0.5)]
    assert keep_tied_at_cut(SCORED, 4) == SCORED[:4]


def test_keep_tied_at_cut_beyond_list_returns_everything():
    """A cut at or beyond the list length returns all items."""
    assert keep_tied_at_cut(SCORED, 5) == SCORED
    assert keep_tied_at_cut(SCORED, 50) == SCORED


@pytest.mark.parametrize("cut", [0, -1])
def test_keep_tied_at_cut_raises_value_error(cut):
    """cut of zero or below is rejected with a ValueError."""
    with pytest.raises(ValueError):
        keep_tied_at_cut(SCORED, cut)


def test_prefer_vector_side_orders_equal_scores_by_vector_membership():
    """Among equal scores, ids from the vector side come first."""
    scored = [('c', 0.3), ('a', 0.2), ('b', 0.2)]
    assert prefer_vector_side(scored, ['b']) == [('c', 0.3), ('b', 0.2), ('a', 0.2)]


def test_prefer_vector_side_falls_back_to_id_order():
    """Equal scores with no vector-side difference stay in id order."""
    scored = [('a', 0.2), ('b', 0.2)]
    assert prefer_vector_side(scored, []) == [('a', 0.2), ('b', 0.2)]
    assert prefer_vector_side(scored, ['a', 'b']) == [('a', 0.2), ('b', 0.2)]


def test_prefer_vector_side_does_not_override_scores():
    """A vector-side id with a lower score must not jump above a higher score."""
    scored = [('c', 0.3), ('a', 0.2)]
    assert prefer_vector_side(scored, ['a']) == [('c', 0.3), ('a', 0.2)]


def test_keep_tied_at_cut_all_tied_to_the_end_of_the_list():
    """If every item after the cut ties with the last kept one, all of them are kept."""
    scored = [('a', 0.4), ('b', 0.4), ('c', 0.4)]
    assert keep_tied_at_cut(scored, 1) == scored


def test_prefer_vector_side_orders_remaining_ties_by_id():
    """Equal scores on the same side are ordered by id, whatever the input order."""
    scored = [('b', 0.2), ('a', 0.2), ('d', 0.2), ('c', 0.2)]
    assert prefer_vector_side(scored, ['c', 'd']) == [('c', 0.2), ('d', 0.2), ('a', 0.2), ('b', 0.2)]
