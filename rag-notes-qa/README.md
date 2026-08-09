# rag-notes-qa

I'm setting up the data ingestion and chunking pipeline for a RAG system. 
It's built on top of my personal AI study notes, specifically the `.docx` files 
inside `ai-study/week-*` and `AI-Study-Comprehensive-Notes.docx` file. 
This covers Phase 3 of the project. Moving forward, the later weeks will focus on 
adding embeddings, vector search, hybrid retrieval, a citation-linked CLI, 
and testing everything against a hand-labeled golden dataset.

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
- `src/rag_notes/models.py` — `DocumentMetadata`, `SourceDocument`, `Chunk` dataclasses
- `src/rag_notes/structure_chunker.py` — splits a document along its own heading
  boundaries (`chunk_document`)
- `src/rag_notes/fixed_chunker.py` — splits raw text into fixed-size, overlapping
  chunks by token count (`chunk_fixed_size`, `chunk_fixed_size_document`)
- `data/golden_qa.json` — hand-labeled (question, source document, verbatim answer
  span) reference set, chunker-agnostic and retrieval-mode-agnostic by design
- `tests/` — unit tests for both chunkers
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
dataset, packaging, and tests. Retrieval, embeddings, and the RAG CLI come in later
weeks.