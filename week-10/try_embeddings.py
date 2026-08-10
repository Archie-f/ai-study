from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_notes.embedder import load_embedding_model, embed_chunks, cosine_similarity
from rag_notes.loader import load_corpus
from rag_notes.structure_chunker import chunk_document, BOUNDARY_STYLES

text = "Python type hints are optional at runtime."
sentences = [
    "Python type hints are optional at runtime.",
    "Static type annotations in Python don't change how the code runs.",
    "The espresso machine needs to be descaled every three months.",
]
model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode(text)

print(type(vector))
print(vector.shape)
print(len(vector))
print(vector[:5])

vectors = model.encode(sentences)
similarity_1 = cosine_similarity(vectors[0], vectors[1])
similarity_2 = cosine_similarity(vectors[0], vectors[2])
similarity_3 = cosine_similarity(vectors[1], vectors[2])
print(f"Similarity-1: {similarity_1}")
print(f"Similarity-2: {similarity_2}")
print(f"Similarity-3: {similarity_3}")

# load -> chunk -> embed -> compare
# 1. load
documents = load_corpus(Path("/Users/archie/Documents/Claude/Projects/AI Study"))
week01_day01 = documents[0]

# 2. chunk
chunks = chunk_document(week01_day01, BOUNDARY_STYLES)

# 3. embed
model = load_embedding_model()
embedded_chunks = embed_chunks(chunks, model)

query = "Why do type hints matter in Python even though the language doesn't enforce them at runtime?"
query_vector = model.encode(query)

# 4. compare
max_similar = max(
    ((cosine_similarity(np.array(query_vector), np.array(ec.vector)), ec.chunk.heading) for ec in embedded_chunks),
    key=lambda x: x[0],
)
print(f"Max similar: {max_similar}")
