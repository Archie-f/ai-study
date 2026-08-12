from pathlib import Path

import numpy as np

from rag_notes.bm25_index import build_bm25_index, search
from rag_notes.embedder import load_embedding_model, embed_chunks, cosine_similarity
from rag_notes.loader import load_corpus
from rag_notes.structure_chunker import chunk_document, BOUNDARY_STYLES
from rag_notes.vector_store import get_collection, add_chunks

text = "Python type hints are optional at runtime."
sentences = [
    "Python type hints are optional at runtime.",
    "Static type annotations in Python don't change how the code runs.",
    "The espresso machine needs to be descaled every three months.",
]
PATH = str(Path(__file__).parent / "persistent")
model = load_embedding_model()
vector = model.encode(text)

print(type(vector))
print(vector.shape)
print(len(vector))
print(vector[:5])

vectors = model.encode(sentences)
similarity_1 = cosine_similarity(vectors[0], vectors[1])
similarity_2 = cosine_similarity(vectors[0], vectors[2])
similarity_3 = cosine_similarity(vectors[1], vectors[2])
print("--- Cosine similarity exercises")
print(f"Similarity-1: {similarity_1}")
print(f"Similarity-2: {similarity_2}")
print(f"Similarity-3: {similarity_3}")
print()

# load -> chunk -> embed -> vector_store -> query
# 1. load
documents = load_corpus(Path("/Users/archie/Documents/Claude/Projects/AI Study"))
week01_day01 = documents[0]

# 2. chunk
chunks = chunk_document(week01_day01, BOUNDARY_STYLES)

# 3. embed
embedded_chunks = embed_chunks(chunks, model)

# 4. vector store
collection = get_collection(PATH)
add_chunks(collection, embedded_chunks)

# 5. query
query = "Why do type hints matter in Python even though the language doesn't enforce them at runtime?"

## a - dense retrieval
query_vector = model.encode(query)

### exact nearest-neighbor search (compares manually with a loop to find the max_similar. !!!Not included in the RAG pipeline!!!)
max_similar = max(
    ((cosine_similarity(np.array(query_vector), np.array(ec.vector)), ec.chunk.heading) for ec in embedded_chunks),
    key=lambda x: x[0],
)
print("--- Dense retrieval - max similar (manual loop)")
print(f"Max similar: {max_similar}")
print()

### approximate nearest-neighbor search
results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=3,
    include=["metadatas", "documents", "distances"],
)
print("--- Dense retrieval - approximate nearest-neighbor search")
for chunk_id, distance, metadata in zip(
    results["ids"][0], results["distances"][0], results["metadatas"][0]
):
    print(f"{distance:.3f}  {metadata['heading']}  ({chunk_id})")
print()

## sparse retrieval
bm25_index = build_bm25_index(chunks)
scored_results = search(query, bm25_index, n_results=3)

print("--- Sparse retrieval - BM25 search")
for score, chunk in scored_results:
    print(f"{score:.3f}  {chunk.heading}")
