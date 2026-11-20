from typing import TypedDict


class CartState(TypedDict):
    items: list[tuple[float, int]]   # (unit_price, quantity)
    subtotal: float
    total: float

def compute_subtotal(state: CartState) -> dict:
    """Add up unit_price * quantity over all items.

    Returns only the key it changes.
    Example: items [(10.0, 2), (5.0, 1)] -> {"subtotal": 25.0}
    Example: items []                    -> {"subtotal": 0.0}
    """
    subtotal = sum(p * q for p, q in state["items"])
    return {"subtotal": subtotal}

def apply_discount(state: CartState) -> dict:
    """Take 10% off when subtotal is 20.0 or more; otherwise change nothing.

    Reads state['subtotal'], returns only the key 'total'.
    Examples: subtotal 25.0 -> {"total": 22.5}
              subtotal 8.0  -> {"total": 8.0}
              subtotal 20.0 -> {"total": 18.0}
    """
    subtotal = state['subtotal']
    if subtotal >= 20.0:
        return {"total": 0.9 * subtotal}
    else:
        return {"total": subtotal}
