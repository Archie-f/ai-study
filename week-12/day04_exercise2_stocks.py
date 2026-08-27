class OutOfStockError(Exception):
    ...  # raised when requested qty exceeds available stock


class Reservation:
    pass


def reserve_item(sku: str, qty: int) -> Reservation:
    ...  # returns a Reservation, or raises OutOfStockError


def safe_reserve(sku: str, qty: int) -> Reservation | None:
    """Reserve stock, returning None instead of raising if unavailable.

    Args:
        sku: The item to reserve.
        qty: The quantity requested.

    Returns:
        The Reservation if stock was available, otherwise None.
    """
    try:
        return reserve_item(sku, qty)
    except OutOfStockError:
        return None