class LookupError(Exception):
    ...  # raised when no matching record exists


class CustomerRecord:
    pass


def lookup_customer(customer_id: str) -> CustomerRecord:
    ...  # returns a CustomerRecord, or raises LookupError if none found


def safe_resolve_customer(customer_id: str) -> CustomerRecord | None:
    """Look up a customer, returning None instead of raising if not found.

    Args:
        customer_id: The customer to look up.

    Returns:
        The CustomerRecord if found, otherwise None.
    """
    try:
        return lookup_customer(customer_id)
    except LookupError:
        return None