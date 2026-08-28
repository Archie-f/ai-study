def format_price(cents: int) -> str:
    ...  # e.g. 1050 -> "$10.50", 0 -> "$0.00", -250 -> "-$2.50"


def build_price_test_cases() -> list[tuple[int, str]]:
    """Build a set of (cents, expected_formatted_price) test cases.

    Returns:
        A list covering at least a whole-dollar amount, a fractional
        amount, zero, and a negative amount (a refund).
    """
    return [
        (3000, "$3.00"),
        (1050, "$10.50"),
        (0, "$0.00"),
        (-250, "-$2.50")
    ]