# rag-notes-qa

I'm setting up the data ingestion, chunking, and retrieval pipeline for a RAG system.
It's built on top of my personal AI study notes, specifically the `.docx` files
inside `ai-study/week-*` and `AI-Study-Comprehensive-Notes.docx` file.
This covers Phase 3 of the project. Week 9 handled ingestion and chunking; Week 10
adds embeddings, a Chroma vector store, BM25 sparse search, and hybrid retrieval via
Reciprocal Rank Fusion, wired into one retrieval module. A citation-linked CLI
comes in a later week.

## Setup

From inside `rag-notes-qa/`, using the shared `ai-study/.venv`:  
```bash
source ../.venv/bin/activate  
pip install -e .  
```
This installs `rag_notes` as an editable package, so `pytest` and any script here
import it correctly from any directory — no `PYTHONPATH` needed.

Copy `.env.example` to `.env` and set `NOTES_ROOT` to the path of the notes corpus
(the `ai-study` folder containing the `week-*` subfolders).

## Structure

- `src/rag_notes/loader.py` — walks `NOTES_ROOT`, loads in-scope `.docx` files into
  `SourceDocument` objects (raw paragraphs + style names)
- `src/rag_notes/models.py` — `DocumentMetadata`, `SourceDocument`, `Chunk`,
  `EmbeddedChunk`, `BM25Index`, `RetrievalIndex` dataclasses
- `src/rag_notes/structure_chunker.py` — splits a document along its own heading
  boundaries (`chunk_document`)
- `src/rag_notes/fixed_chunker.py` — splits raw text into fixed-size, overlapping
  chunks by token count (`chunk_fixed_size`, `chunk_fixed_size_document`)
- `src/rag_notes/embedder.py` — loads a local sentence-transformers model and embeds
  chunks (`load_embedding_model`, `embed_chunks`)
- `src/rag_notes/vector_store.py` — opens/creates, deletes, and populates a Chroma
  collection, and runs dense queries against it (`get_collection`,
  `delete_collection`, `add_chunks`, `get_query_result`)
- `src/rag_notes/bm25_index.py` — builds a BM25 sparse index and scores documents
  against a query (`build_bm25_index`, `score_document`, `bm25_search`)
- `src/rag_notes/hybrid_search.py` — merges dense and sparse results with
  Reciprocal Rank Fusion (`hybrid_search`)
- `src/rag_notes/retrieval.py` — the single entry point: `build_retrieval_index()`
  loads, chunks, embeds, and indexes the whole corpus; `search()` runs a hybrid
  query against the result
- `data/golden_qa.json` — hand-labeled (question, source document, verbatim answer
  span) reference set, chunker-agnostic and retrieval-mode-agnostic by design
- `tests/` — unit tests for both chunkers and for the pure functions in
  `bm25_index.py` and `hybrid_search.py`
- `compare_chunkers.py` — runs both chunkers over one document and prints their
  output side by side
- `main.py` — loads the full corpus and structure-chunks every document

## Running things
```bash
pytest # run the test suite
python compare_chunkers.py # compare structure-aware vs. fixed-size chunking on one doc
python main.py # structure-chunk the whole corpus
```

## Status

Week 09 (ingestion + chunking) complete: corpus loader, both chunkers, golden Q&A
dataset, packaging, and tests.

Week 10 is officially wrapped. I've put together `retrieval.py` to handle the whole pipeline 
using embeddings, vector stores, BM25, and hybrid search through `build_retrieval_index()` 
and `search()`.
On the testing side, I made sure all the pure functions like `build_rank_map`, `rrf_merge`, 
`tokenize`, `build_doc_frequencies`, and `score_document` are fully covered by unit tests. 
`embed_chunks()` and `get_query_result()` are missing from 
the fast unit test suite. That’s a deliberate choice, not an oversight, 
since they actually require a live sentence-transformers model and a real Chroma 
collection to run.  
**Heads up:** `main.py` is still running the older Week 09 chunking setup for now. 
Hooking up the CLI to the new search() function in retrieval.py for an end-to-end flow is 
0next up on the to-do list.