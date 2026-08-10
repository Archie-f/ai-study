from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")
reviews = [
        "The battery life on this laptop is incredible, easily lasts two full days.",
        "Package arrived damaged and customer service never responded.",
        "Great value for the price, would buy again.",
    ]

def embed_reviews(reviews: list[str]) -> None:
    """Load the default embedding model, embed each review, and print
    its vector's shape and first 3 values.

    Args:
        reviews: raw review text strings
    """
    vectors = model.encode(reviews)
    for vector in vectors:
        print(f"{vector.shape} / {vector[:3]}")
        print("-" * 20)


if __name__ == "__main__":
    embed_reviews(reviews)

