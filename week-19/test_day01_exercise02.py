import pytest

from day01_exercise02_cart_nodes import compute_subtotal, apply_discount ,CartState

state_normal: CartState = {"items": [(10.0, 2), (5.0, 1)], "subtotal": 0.0, "total": 0.0}
state_below_20: CartState = {"items": [(4.0, 2)], "subtotal": 0.0,  "total": 0.0}
state_edge_case: CartState = {"items": [(10.0, 2)], "subtotal": 20.0, "total": 0.0}
state_empty_items: CartState = {"items": [], "subtotal": 0.0, "total": 0.0}


def test_result_dicts() -> None:
    computed_subtotal = compute_subtotal(state_normal)
    assert computed_subtotal  == {"subtotal": 25.0}
    state_normal["subtotal"] = compute_subtotal(state_normal)["subtotal"]
    assert apply_discount(state_normal) == pytest.approx({"total": 22.5})


def test_normal_subtotal() -> None:
    state_normal["subtotal"] = compute_subtotal(state_normal)["subtotal"]
    assert apply_discount(state_normal) == pytest.approx({"total": 22.5})


def test_below_20() -> None:
    state_below_20["subtotal"] = compute_subtotal(state_below_20)["subtotal"]
    assert apply_discount(state_below_20) == pytest.approx({"total": 8.0})


def test_edge_case() -> None:
    state_edge_case["subtotal"] = compute_subtotal(state_edge_case)["subtotal"]
    assert apply_discount(state_edge_case) == pytest.approx({"total": 18.0})


def test_empty_items() -> None:
    assert compute_subtotal(state_empty_items) == pytest.approx({"subtotal": 0.0})
