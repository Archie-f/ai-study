import math
import re
from collections import Counter

from rag_notes.models import Chunk, BM25Index


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into word tokens, dropping
    punctuation.

    Args:
        text: raw text
    Returns:
        list of lowercase word tokens
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def build_doc_frequencies(tokenized_docs: list[list[str]]) -> Counter:
    """Count how many documents each term appears in at least once.

    Args:
        tokenized_docs: each document already split into tokens
    Returns:
        Counter mapping term -> number of documents containing it
    """
    frequency = Counter()
    for tokenized_doc in tokenized_docs:
        frequency.update(set(tokenized_doc))

    return frequency


def build_bm25_index(chunks: list[Chunk]) -> BM25Index:
    """Build a BM25 index over a list of Chunks.

    Args:
        chunks: chunks produced by either chunker
    Returns:
        a BM25Index ready for search()
    """
    tokenized_docs = [tokenize(chunk.text) for chunk in chunks]
    doc_freq = build_doc_frequencies(tokenized_docs)
    avgdl = sum(len(tokenized_doc) for tokenized_doc in tokenized_docs) / len(tokenized_docs)

    return BM25Index(
        chunks=chunks,
        tokenized_docs=tokenized_docs,
        doc_freq=doc_freq,
        avgdl=avgdl,
    )


def score_document(query_terms: list[str], doc_index: int, index: BM25Index) -> float:
    """Compute the BM25 score of one document against a tokenized
    query.

    Args:
        query_terms: tokenized query
        doc_index: which document in index.chunks to score
        index: a built BM25Index
    Returns:
        BM25 score (higher is more relevant; 0.0 if no terms match)
    """
    doc_tokens = index.tokenized_docs[doc_index]
    doc_len = len(doc_tokens)
    n_docs = len(index.chunks)

    score = 0.0
    for term in query_terms:
        n_term = index.doc_freq.get(term, 0)
        if n_term == 0:
            continue  # term never appears anywhere in the corpus
        idf = math.log((n_docs - n_term + 0.5) / (n_term + 0.5) + 1)
        f = doc_tokens.count(term)
        numerator = f * (index.k1 + 1)
        denominator = f + index.k1 * (1 - index.b + index.b * doc_len / index.avgdl)
        score += idf * (numerator / denominator)
    return score


def search(query: str, index: BM25Index, n_results: int = 3) -> list[tuple[float, Chunk]]:
    """Score a query against every document in the index and return
    the top matches.

    Args:
        query: raw query string
        index: a built BM25Index
        n_results: how many top matches to return
    Returns:
        list of (score, chunk) tuples, sorted highest score first
    """
    query_terms = tokenize(query)
    scored = [
        (score_document(query_terms, i, index), index.chunks[i])
        for i in range(len(index.chunks))
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:n_results]