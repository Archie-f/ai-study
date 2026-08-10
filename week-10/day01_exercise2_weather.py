from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

forecasts = [
    "Heavy rain expected across the coast this afternoon, with possible flooding.",
    "Showers will move in by evening, so bring an umbrella.",
    "Clear skies and sunshine all day, perfect for a hike.",
]


def rank_similarities(sentences: list[str]) -> list[tuple[str, str, float]]:
    """Compute cosine similarity for every pair of sentences and return
    them sorted, most similar first.

    Args:
        sentences: at least two raw sentence strings
    Returns:
        list of (sentence_a, sentence_b, similarity) tuples, sorted
        descending by similarity
    """
    similarity_list: list[tuple[str, str, float]] = []
    vectors = model.encode(sentences)

    for (sen1, a), (sen2, b) in combinations(zip(sentences, vectors), 2):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        similarity = float(dot_product / (norm_a * norm_b))
        similarity_list.append((sen1, sen2, similarity))

    return sorted(similarity_list, key=lambda x: x[2], reverse=True)

if __name__ == "__main__":
    result = rank_similarities(forecasts)
    print(len(result))
    print(result)