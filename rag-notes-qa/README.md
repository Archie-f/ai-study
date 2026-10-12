# rag-notes-qa

I'm setting up the data ingestion, chunking, and retrieval pipeline for a RAG system.
It's built on top of my personal AI study notes, specifically the `.docx` files
inside `ai-study/week-*` and `AI-Study-Comprehensive-Notes.docx` file.
This covers Phase 3 of the project.

Week 9 handled ingestion and chunking; 

Week 10 adds embeddings, a Chroma vector store, BM25 sparse search, and hybrid retrieval via
Reciprocal Rank Fusion, wired into one retrieval module. 

Week 11 was all about consolidation. Instead of writing new pipeline code, 
I rebuilt the chunkers, embedder, and retrieval modules completely from memory 
and actually caught and backported two genuine bugs in the process. 
I also spent time reviewing some Python fundamentals and did a deep dive reading pass 
over LangChain and LlamaIndex to see how they stacked up against my own project's design.

Week 12 finally brings the full pipeline together. I wrapped up prompt assembly, 
generation, structured citations, and a solid guardrail to block unsupported answers. 
I built a CLI so you can finally ask questions against the entire corpus end-to-end.

Week 13 adds evaluation: retrieval-quality metrics (recall@k, MRR), an LLM-as-judge
faithfulness scorer, and a combined HTML report that ties both together with a
filterable, per-question results table.

Week 14 was the second consolidation week: generation and evaluation rebuilt from
memory, plus three swappable interfaces (`Chunker`, `VectorStore`, `Retriever`)
and their adapters.

Week 15 wires those interfaces into the real pipeline and uses them to measure:
a frozen evaluation corpus, a chunking-strategy comparison, two retrieval fixes
that made the evaluation reproducible, and a retrieval-mode comparison.

## Setup

From inside `rag-notes-qa/`, using the shared `ai-study/.venv`:  
```bash
source ../.venv/bin/activate  
pip install -e .  
```
This installs `rag_notes` as an editable package, so `pytest` and any script here
import it correctly from any directory — no `PYTHONPATH` needed.

Copy `.env.example` to `.env` and set `NOTES_ROOT` to the path of the notes corpus
(the `ai-study` folder containing the `week-*` subfolders). `NOTES_ROOT` is only
needed by `ask.py` and the demo scripts in `verify/`; the evaluation scripts read
the frozen corpus in `data/eval_corpus/` instead.

All script-side paths live in one place, `config.py` (`PERSIST_PATH`,
`GOLDEN_QA_PATH`, `EVAL_CORPUS_PATH`, and `get_notes_root()`). The `rag_notes`
package itself never reads paths or environment variables — callers pass them in.

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
  against a query (`build_bm25_index`, `score_document`, `bm25_search`);
  `bm25_search()` returns only chunks that contain at least one query term
  (at most `n_results`, possibly none)
- `src/rag_notes/hybrid_search.py` — merges dense and sparse results with
  Reciprocal Rank Fusion (`hybrid_search`); equal RRF scores are ordered by
  chunk id, so results never depend on set iteration order
- `src/rag_notes/retrieval.py` — the single entry point:
  `build_retrieval_index(notes_root, vector_store, chunker)` loads, chunks,
  embeds, and indexes the whole corpus with whatever `Chunker` and `VectorStore`
  it is given (it empties the store first via `reset()`); `search()` runs a query
  in `hybrid`, `vector`, or `bm25` mode
- `src/rag_notes/protocols.py` — structural-typing interfaces for the three
  pieces of the pipeline meant to be swappable: `Chunker`, `VectorStore`,
  `Retriever`
- `src/rag_notes/adapters.py` — concrete adapters implementing those protocols
  (`StructureChunker`, `FixedSizeChunker`, `ChromaVectorStore`,
  `HybridRetriever`), each a thin wrapper around an already-existing
  function/module
- `src/rag_notes/generate.py` — assembles retrieved chunks into a labeled context
  (`build_context`), calls the LLM provider (`generate_answer`), enforces the
  guardrail against unsupported answers (`normalize_answer`), and runs the full
  pipeline end-to-end (`answer_question(retriever, question, provider, n)`,
  which takes any `Retriever`)
- `src/rag_notes/citations.py` — turns search results into structured, numbered
  citations (`build_source_label`, `build_citations`, `format_citations`)
- `src/rag_notes/retrieval_metrics.py` — pure retrieval-quality metrics
  (`recall_at_k`, `reciprocal_rank`, `mean_reciprocal_rank`)
- `src/rag_notes/llm_judge.py` — LLM-as-judge faithfulness scoring
  (`build_judge_prompt`, `clean_result`, `normalize_score`, `judge_answer`)
- `src/rag_notes/eval_report.py` — builds the self-contained HTML eval report
  (`build_full_eval_html`, `save_full_eval_html`)
- `config.py` — script-side configuration: shared paths and `get_notes_root()`
- `ask.py` — CLI entry point: asks a question, prints the answer with its
  citations (`display_answer`)
- `verify/` — manual, live end-to-end smoke tests against a real Ollama model and
  a real Chroma index (`verify_generate.py`, `verify_citations.py`,
  `verify_retrieval.py`, `verify_guardrail.py`, `verify_retrieval_eval.py`,
  `verify_chunking_eval.py`, `verify_llm_judge.py`, `verify_full_eval.py`) —
  distinct from `tests/`, which is fast and dependency-free. The evaluation
  scripts (`*_eval.py`, `verify_llm_judge.py`) build from the frozen corpus; the
  demo scripts build from the live notes under `NOTES_ROOT`
- `data/golden_qa.json` — hand-labeled (question, source document, verbatim answer
  span) reference set, chunker-agnostic and retrieval-mode-agnostic by design
- `data/eval_corpus/` — frozen evaluation corpus: Weeks 01–14 takeaway notes plus
  the Comprehensive Notes as of 2026-10-07 (71 documents). The `.docx` files are
  gitignored; `MANIFEST.txt` lists them. Refreshed only deliberately, together
  with a new baseline
- `tests/` — unit tests for both chunkers, the pure functions in `bm25_index.py`
  and `hybrid_search.py`, `citations.py` (`test_citations.py`), and
  `generate.py`'s context builder and guardrail logic (`test_generate.py`,
  `test_guardrail.py`), plus `retrieval_metrics.py` (`test_retrieval_metrics.py`),
  `llm_judge.py`'s pure pieces (`test_llm_judge.py`), and `eval_report.py`'s
  `build_full_eval_html()` (`test_eval_report.py`), `answer_question()` with fake
  retriever/provider (`test_generate.py`), `hybrid_search()` in all three modes
  with a fake vector store, RRF tie-breaking and zero-score filtering
  (`test_hybrid_search.py`, `test_bm25_index.py`), and the chunking-comparison
  table (`test_chunking_eval.py`)
- `compare_chunkers.py` — runs both chunkers over one document and prints their
  output side by side
- `main.py` — loads the full corpus and structure-chunks every document

**A known trade-off:** `protocols.py` imports `QueryResult` from `chromadb`
and `SentenceTransformer` from `sentence-transformers` for `VectorStore`'s
type hints. That means the "pure interface" file isn't fully independent of
a specific implementation — swapping `ChromaVectorStore` for a different
vector store later would mean touching the protocol's return type too.
Noted here on purpose, not fixed, since a store-agnostic result type is out
of scope for now.

## Running things
```bash
pytest # run the test suite
python compare_chunkers.py # compare structure-aware vs. fixed-size chunking on one doc
python main.py # structure-chunk the whole corpus
python -m verify.verify_retrieval_eval   # recall@k / MRR for all three modes (frozen corpus)
python -m verify.verify_chunking_eval    # structure-aware vs. fixed-size chunking (frozen corpus)
python ask.py --question "What is RRF?"  # end-to-end answer over the live notes
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

Week 12 is wrapped: `generate.py` (context assembly, generation, the
`NO_ANSWER_TOKEN`/`GUARDRAIL_ANSWER` guardrail), `citations.py` (structured,
numbered citations), and `ask.py` (the end-to-end CLI) are all in place and
tested — `tests/test_generate.py`, `tests/test_citations.py`, and
`tests/test_guardrail.py` cover every pure function; the guardrail itself is
additionally exercised live against a real model via
`verify/verify_guardrail.py`, including an adversarial prompt-injection case.
Tagged `v0.12.0`.

Week 13 built out evaluation: `retrieval_metrics.py` (`recall_at_k`,
`reciprocal_rank`, `mean_reciprocal_rank`), `llm_judge.py` (an LLM-as-judge
faithfulness scorer via `judge_answer()`), and `eval_report.py`, which combines
both into one self-contained HTML report with a filterable, per-question
results table. `verify/verify_retrieval_eval.py`, `verify/verify_llm_judge.py`,
and `verify/verify_full_eval.py` run the real pipeline against the corpus and a
real judge model. On the testing side, `tests/test_retrieval_metrics.py`,
`tests/test_llm_judge.py`, and `tests/test_eval_report.py` cover every pure
function — `judge_answer()`, `run_retrieval_eval()`, `compare_retrieval_modes()`,
`run_full_judge_eval()`, and `verify_full_eval.main()` are deliberately excluded
from the fast unit test suite, the same live-call boundary Week 10 drew around
`embed_chunks()` and `get_query_result()`, since they require a real Chroma
index and a real LLM call to run.
Tagged `v0.13.0`.

Week 15 wired the Week 14 interfaces into the pipeline: `build_retrieval_index()`
takes a `Chunker` and a `VectorStore`, `hybrid_search()` takes any `VectorStore`,
and `answer_question()` takes any `Retriever` — inside `src/rag_notes/` only
`vector_store.py` and `adapters.py` still know about Chroma. These are breaking
signature changes compared with `v0.14.0`.

Measurement work on the frozen corpus (StructureChunker, 13 golden questions):

- Chunking comparison: fixed-size chunking scored higher than structure-aware,
  but a per-document breakdown showed the gain follows chunk counts (the
  Comprehensive Notes went from 26 to 90 chunks), not the strategy. Because
  relevance is judged per document, the metric rewards more chunks per relevant
  document — the experiment does not show that fixed-size chunking is better.
- Reproducibility: hybrid results used to change between runs (RRF ties broken
  by set iteration order); now ties are broken by chunk id, and recall is
  identical across runs. A rare residual MRR swing (about 1–2 runs in 16) is
  still under investigation.
- `bm25_search()` no longer pads its results with zero-score chunks.
- Retrieval modes: hybrid recall@1/3/5 = 0.31/0.54/0.54, MRR 0.40; vector
  0.38/0.38/0.54, MRR 0.42; bm25 0.31/0.38/0.46, MRR 0.35. No consistent winner
  between hybrid and vector on 13 questions; a weak but consistent edge for
  hybrid over bm25.

Tagged `v0.15.0` (MINOR: new capability plus breaking signature changes, which
bump MINOR while the version is below 1.0.0).
