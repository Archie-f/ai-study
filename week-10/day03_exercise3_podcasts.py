import math

from rag_notes.bm25_index import tokenize, build_doc_frequencies


def find_matching_episodes(query: str, titles: list[str]) -> str:
    """Build a BM25 index over podcast episode titles and return the
    single best match to query.

    Args:
        query: a listener's raw search phrase
        titles: episode title strings
    Returns:
        the title with the highest BM25 score
    """
    query_terms = tokenize(query)

    tokenized_titles = [tokenize(title) for title in titles]
    doc_freq = build_doc_frequencies(tokenized_titles)
    avgdl = sum(len(tokenized_title) for tokenized_title in tokenized_titles) / len(tokenized_titles)
    n_docs = len(titles)

    scored = []
    k1 = 1.2
    b = 0.75
    for i in range(len(tokenized_titles)):
        doc_tokens = tokenized_titles[i]
        score = 0.0
        for term in query_terms:
            n_term = doc_freq.get(term, 0)
            if n_term == 0:
                continue
            idf = math.log((n_docs - n_term + 0.5) / (n_term + 0.5) + 1)
            f = doc_tokens.count(term)
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * len(doc_tokens) / avgdl)
            score += idf * (numerator / denominator)
        scored.append((score, titles[i]))

    sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)
    print(sorted_scored)
    return sorted_scored[0][1]


titles = [
    "Episode 42: How Ancient Rome Financed Its Wars",
    "Episode 43: A Beginner's Guide to Sourdough Bread",
    "Episode 44: The Economics of Roman Aqueducts",
]
query = "Roman military spending and war finance"


if __name__ == "__main__":
    print(find_matching_episodes(query, titles))