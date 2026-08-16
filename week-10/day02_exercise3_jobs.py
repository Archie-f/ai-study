from pathlib import Path

from rag_notes.embedder import load_embedding_model
from rag_notes.vector_store import get_collection


def find_best_job_match(resume_summary: str, postings: list[dict]) -> tuple[str, float]:
    """Embed and store job postings in a fresh Chroma collection, then
    return the title and distance of the closest match to resume_summary.

    Args:
        resume_summary: text describing a candidate's background
        postings: list of dicts with keys "title" and "description"
    Returns:
        (title, distance) of the closest-matching posting
    """
    model = load_embedding_model()
    # load -> chunk -> embed
    posting_descriptions = [posting['description'] for posting in postings]
    vectors = model.encode(posting_descriptions)

    # vector_store
    collection = get_collection(str(Path(__file__).parent / 'persistent'), name="job_postings_collection")
    ids = [f"posting-{index}" for index in range(len(postings))]
    collection.add(
        ids=ids,
        embeddings=vectors,
        metadatas=postings,
        documents=posting_descriptions,
    )

    # query
    query_vector = model.encode(resume_summary)
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=1,
        include=["metadatas","documents", "distances"]
    )
    print(f"Distance: {result['distances'][0][0]:.3f}")
    print(f"Title: {result['metadatas'][0][0]['title']}")
    return result['metadatas'][0][0]['title'], result['distances'][0][0]


postings = [
    {"title": "Backend Engineer", "description": "Building and scaling REST APIs in Python and Go."},
    {"title": "Data Analyst", "description": "SQL reporting and dashboarding for the sales team."},
    {"title": "ML Engineer", "description": "Training and deploying embedding and retrieval models."},
]
resume_summary = "Experienced building RAG pipelines with sentence-transformers and vector search."

if __name__ == "__main__":
    find_best_job_match(resume_summary, postings)
