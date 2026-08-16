import numpy as np

from rag_notes.embedder import load_embedding_model


def find_most_similar(query: str, candidates: list[str]) -> str:
    """Return the single candidate string most semantically similar
    to query.

    Args:
        query: text to compare against
        candidates: candidate strings to search among
    Returns:
        the candidate string with the highest cosine similarity to query
    """

    model = load_embedding_model()
    candidates_vectors = model.encode(candidates)
    query_vector = np.array(model.encode(query))
    results: list[tuple[float, str]] = []

    for vector, candidate in zip(candidates_vectors, candidates):
        dot_product = np.dot(query_vector, vector)
        norm_query = np.linalg.norm(query_vector)
        norm_vector = np.linalg.norm(vector)
        similarity = float(dot_product / (norm_query * norm_vector))
        results.append((similarity, candidate))

    return sorted(results, key=lambda x: x[0], reverse=True)[0][1]


tickets = [
    "Can't log into my account, password reset email never arrives.",
    "App crashes every time I try to upload a photo.",
    "How do I cancel my subscription before the renewal date?",
]
query = "I forgot my password and the reset link isn't showing up in my inbox."

if __name__ == "__main__":
    print(find_most_similar(query, tickets))