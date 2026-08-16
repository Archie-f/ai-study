def term_frequencies(term: str, documents: list[str]) -> list[int]:
    """Count how many times term appears in each document, after
    lowercasing both the term and each document's words.

    Args:
        term: a single query word
        documents: raw document strings
    Returns:
        one count per document, same order as documents
    """
    counts = []

    for document in documents:
        text = document.lower().split()
        counts.append(text.count(term))

    return counts


headlines = [
    "Central bank raises interest rates for the third time this year",
    "Local team wins championship after dramatic final rate of scoring",
    "Interest in renewable energy rates highest growth this year",
]
term = "rate"


if __name__ == '__main__':
    counts = term_frequencies(term, headlines)
    print(counts)
