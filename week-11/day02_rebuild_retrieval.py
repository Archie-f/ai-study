import math
import os
import re
from collections import Counter
from pathlib import Path

import chromadb
from chromadb import QueryResult
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag_notes.embedder import embed_chunks, DEFAULT_MODEL_NAME, load_embedding_model
from rag_notes.hybrid_search import hybrid_search
from rag_notes.loader import load_corpus
from rag_notes.models import DocumentMetadata, Chunk, BM25Index, RetrievalIndex
from rag_notes.structure_chunker import BOUNDARY_STYLES, chunk_document


COLLECTION_NAME = "ai_study_notes"


def get_collection(persist_path: str, name: str = COLLECTION_NAME):
    """Open (or create) a cosine-configured Chroma collection on disk."""
    client = chromadb.PersistentClient(persist_path)
    collection = client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}}
    )
    print(f"Collection {name} created")
    return collection


def delete_collection(persist_path, name=COLLECTION_NAME):
    """Delete a collection from disk. Must not crash if it doesn't exist yet."""
    client = chromadb.PersistentClient(persist_path)
    try:
        client.delete_collection(name)
        print(f"Collection {name} deleted")
    except NotFoundError:
        print(f"Collection {name} does not exist")


def build_metadata(chunk) -> dict:
    """Convert a Chunk's fields into a Chroma-safe metadata dict (no None values)."""
    return {
        "source": chunk.source.title,
        "heading": chunk.heading or "",
        "chunk_index": chunk.chunk_index,
    }


def add_chunks(collection, embedded_chunks: list) -> None:
    """Add a batch of EmbeddedChunks to a Chroma collection."""
    ids = [f"{embedded.chunk.source.title}-{embedded.chunk.chunk_index}" for embedded in embedded_chunks]
    collection.add(
        ids=ids,
        embeddings=[embedded.vector for embedded in embedded_chunks],
        metadatas=[build_metadata(embedded.chunk) for embedded in embedded_chunks],
        documents=[embedded.chunk.text for embedded in embedded_chunks],
    )


def get_query_result(
        collection,
        query: str,
        model,
        n_results: int = 3,
        includes=("metadatas", "documents", "distances")
) -> QueryResult:
    """Embed a query string and run collection.query() with it."""
    query_vector = model.encode(query)
    return collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_results,
        include=list(includes),
    )


def tokenize(text: str) -> list[str]:
    """Lowercase text and split into word tokens, dropping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_doc_frequencies(tokenized_docs: list[list[str]]):
    """Count how many DOCUMENTS each term appears in at least once (not total occurrences)."""
    counter = Counter()
    for tokenized_doc in tokenized_docs:
        counter.update(set(tokenized_doc))

    return counter

def build_bm25_index(chunks: list) -> BM25Index:
    """Tokenize every chunk, build doc frequencies, compute avgdl, bundle into a BM25Index."""
    tokenized_chunks = []
    for chunk in chunks:
        tokenized_chunks.append(tokenize(chunk.text))
    frequencies = build_doc_frequencies(tokenized_chunks)
    average_document_length = sum(len(tokenized_doc) for tokenized_doc in tokenized_chunks) / len(tokenized_chunks)

    return BM25Index(
        chunks=chunks,
        tokenized_docs=tokenized_chunks,
        doc_freq=frequencies,
        avgdl=average_document_length,
    )


def score_document(query_terms: list[str], doc_index: int, index: BM25Index) -> float:
    """BM25 score of one document against a tokenized query (k1=1.2, b=0.75)."""
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



def bm25_search(query: str, index: BM25Index, n_results: int = 3) -> list[tuple[float, Chunk]]:
    """Score the query against every document, return the top n_results, highest first."""
    query_terms = tokenize(query)
    results = [
        (score_document(query_terms, i, index), index.chunks[i])
        for i in range(len(index.chunks))
    ]
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:n_results]


def build_rank_map(ordered_ids: list[str]) -> dict[str, int]:
    """Turn an ordered id list into an id -> 1-based rank lookup."""
    return {ordered_id: index for index, ordered_id in enumerate(ordered_ids, start=1)}


def rrf_merge(rank_maps: list[dict[str, int]], k: int = 60) -> list[tuple[str, float]]:
    """Merge multiple id -> rank maps into one RRF-scored, sorted list (highest score first)."""
    rrf_merged = []
    keys_set = set()
    for rank_map in rank_maps:
        keys_set.update(rank_map.keys())

    for key in keys_set:
        score = 0.0
        for rank_map in rank_maps:
            if key not in rank_map:
                continue
            score += float(1 / (k + rank_map[key]))
        rrf_merged.append((key, score))

    rrf_merged.sort(key=lambda x: x[1], reverse=True)
    return rrf_merged


def build_retrieval_index(notes_root: Path, persist_path: str) -> RetrievalIndex:
    """Load, chunk, embed, and index an entire notes corpus, ready for search()."""
    model = load_embedding_model()

    source_documents = load_corpus(notes_root)
    chunks = []
    for source_document in source_documents:
        chunks.extend(chunk_document(source_document, BOUNDARY_STYLES))
    embedded_chunks = embed_chunks(chunks, model)

    delete_collection(persist_path)
    collection = get_collection(persist_path)
    add_chunks(collection, embedded_chunks)

    bm25_index = build_bm25_index(chunks)

    return RetrievalIndex(
        collection=collection,
        model=model,
        bm25_index=bm25_index
    )


def search(retrieval_index: RetrievalIndex, query: str, n: int = 3) -> list[tuple[str, float, Chunk]]:
    """Run a hybrid search against an already-built RetrievalIndex."""
    return hybrid_search(
        query=query,
        collection=retrieval_index.collection,
        model=retrieval_index.model,
        bm25_index=retrieval_index.bm25_index,
        n=n
    )


if __name__ == "__main__":
    path = "week-11/persistent_day02_test"
    metadata = DocumentMetadata(week=1, day=1, file_path=Path("test.docx"), title="test")

    chunks = [
        Chunk(text="Type hints improve code readability.", source=metadata, heading="1.1", chunk_index=0),
        Chunk(text="Cosine similarity measures the angle between two vectors.", source=metadata, heading="2.1",chunk_index=1),
        Chunk(text="The espresso machine needs descaling every three months.", source=metadata, heading=None,chunk_index=2),
        Chunk(text="Vector databases like Chroma store embeddings for fast search.", source=metadata, heading="3.1",chunk_index=3),
    ]
    query = "vector database"

    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    embedded_chunks = embed_chunks(chunks, model)

    delete_collection(path)
    collection = get_collection(persist_path=path, name=COLLECTION_NAME)
    add_chunks(collection=collection, embedded_chunks=embedded_chunks)

    delete_collection(path)
    collection = get_collection(persist_path=path, name=COLLECTION_NAME)
    add_chunks(collection=collection, embedded_chunks=embedded_chunks)

    query_result = get_query_result(collection=collection, query=query, model=model, n_results=2)
    print(query_result)

    print("\n" + "=" * 40 + "\n")

    tokenized_1 = tokenize("Chroma's add() is case-SENSITIVE!")
    tokenized_2 = tokenize("")
    tokenized_3 = tokenize("RAG-2026 pipeline")
    print(tokenized_1)    # Expected: ["chroma", "s", "add", "is", "case", "sensitive"]
    print(tokenized_2)    # Expected: []
    print(tokenized_3)    # Expected: ["rag", "2026", "pipeline"]

    chunks_2 = [
        Chunk(text="the cat sat on the mat", source=metadata, heading=None, chunk_index=0),
        Chunk(text="the dog sat on the log", source=metadata, heading=None, chunk_index=1),
        Chunk(text="cats and dogs are pets", source=metadata, heading=None, chunk_index=2),
    ]

    query_2 = "cat sat"

    bm25_index = build_bm25_index(chunks_2)
    results = bm25_search(query_2, bm25_index)
    for result in results:
        print(f"Score: {result[0]:.3f}, Text: {result[1].text}")

    print("\n" + "=" * 40 + "\n")

    map_1 = ["docA", "docB", "docC"]
    map_2 = ["docB", "docD"]
    empty_map = []
    map_x = ["x"]
    rank_map_1 = build_rank_map(map_1)
    rank_map_2 = build_rank_map(map_2)
    print(rank_map_1)                   # Expected: {"docA": 1, "docB": 2, "docC": 3}
    print(rank_map_2)                   # Expected: {"docB": 1, "docD": 2}
    print(build_rank_map(empty_map))    # Expected: {}
    print(build_rank_map(map_x))        # Expected: {"x": 1}
    print("-" * 40)

    rank_map_list = [rank_map_1, rank_map_2]
    print(rank_map_list)

    rrf_result = rrf_merge(rank_map_list, k=60)
    print(*(f"{k} = {v:.4f}" for k, v in rrf_result), sep="\n")

    # Expected:
    # docB (in both — 1/62 + 1/61 ≈ 0.0325),
    # docA (dense only — 1/61 ≈ 0.0164),
    # docD (sparse only — 1/62 ≈ 0.0161),
    # docC (dense only, worst dense rank — 1/63 ≈ 0.0159)

    print("\n" + "=" * 40 + "\n")

    load_dotenv()
    notes_root = Path(os.getenv("NOTES_ROOT"))
    retrieval_index = build_retrieval_index(notes_root, path)
    search_results = search(retrieval_index, query)
    for search_result in search_results:
        print(f"{search_result[0]} | {search_result[1]:.4f} | {search_result[2].heading}")

