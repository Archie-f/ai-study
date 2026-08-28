NO_MATCH = "NO_MATCH"


def search_catalog(query: str) -> str:
    ...  # returns a product description, or NO_MATCH if nothing matched


def describe_search_result(query: str, result: str) -> str:
    """Describe a catalog search result, or report no match.

    Args:
        query: The search query that produced result.
        result: search_catalog()'s return value — either a product
            description, or the NO_MATCH sentinel.

    Returns:
        result itself when it's a real match, otherwise
        "No match found for <query>".
    """
    if result == NO_MATCH:
        return f"No match found for {query}"
    return result
