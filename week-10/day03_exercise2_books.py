import re

from rag_notes.bm25_index import tokenize, build_doc_frequencies


def build_book_index(descriptions: list[str]) -> dict:
    """Tokenize every book description and build its document-frequency counts.

    Args:
        descriptions: raw book description strings
    Returns:
        dict with keys "tokenized_docs" (list[list[str]]) and
        "doc_freq" (Counter)
    """
    tokenized_docs = [tokenize(description) for description in descriptions]
    frequency = build_doc_frequencies(tokenized_docs)

    return {"tokenized_docs": tokenized_docs, "doc_freq": frequency}


descriptions = [
    "A young wizard discovers his magical heritage and attends a school of witchcraft.",
    "A detective investigates a murder aboard a luxury train stuck in the snow.",
    "A hobbit journeys across a dangerous land to destroy a powerful ring.",
]


if __name__ == "__main__":
    print(build_book_index(descriptions)["doc_freq"])