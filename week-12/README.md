# Week 12 — Retrieval-Augmented Generation: Prompt Assembly, Citations, Guardrails, CLI

Week 12 closes out the RAG pipeline started in Week 09: after ingestion/chunking (Week 09)
and hybrid retrieval (Week 10), this week adds the final two stages — prompt assembly +
generation, and structured citations — plus a guardrail against unsupported answers and
a real end-to-end CLI. Reuses Phase 2's `LLMProvider` abstraction (`AnthropicProvider`,
`OllamaProvider`) as the generation step rather than reinventing it.

## Day-by-Day

**Day 01 (2026-08-25) — `7c71e0a`**
Built `rag-notes-qa/src/rag_notes/generate.py`: `build_context()`, `SYSTEM_PROMPT`,
`generate_answer()`. `verify_generate.py` end-to-end tested the full pipeline against the
real corpus and a live local `llama3`; the guardrail held for an out-of-scope question.
Real bugs found and fixed: a trailing-blank-line string-building bug (fixed with
list + `join`), an `and`/`or` logic bug in the metadata fallback, an exercise's
system/user-prompt inversion, a missing "Question:" label.

**Day 02 (2026-08-25) — `964b9ef`**
Built `citations.py` — `build_source_label()`, `Citation`/`AnsweredQuery` dataclasses,
`build_citations()`, `format_citations()`. `tests/test_citations.py` (10 unit tests, no
live dependency) — the first step toward closing Week 11's test-coverage gap.
`verify_citations.py` confirmed citation numbering matches `build_context()`'s exactly.
Real bugs: an off-by-one in `build_citations()`'s enumeration (missing `start=1`),
`format_citations()` not guarding against `heading=None`, an unrelated pre-existing
missing-space bug in `generate_answer()`'s "Question:" label.

**Day 03 (2026-08-26) — `62625d4`**
Built `answer_question(index, question, provider, n) -> AnsweredQuery`, the single
orchestration function replacing duplicated `search() → generate_answer() →
build_citations()` wiring previously hand-typed in two verify scripts, plus
`display_answer()`. Built `ask.py` — an argparse CLI (`--question`, `--n`) matching the
project's existing convention. Real bugs: the guardrail's own detection was fragile
(string-matching "I don't know" produced both false positives and false negatives, traced
to Ollama sampling non-determinism) — fixed by introducing a `NO_ANSWER_TOKEN` sentinel
the system prompt emits verbatim, decoupled from the human-facing `GUARDRAIL_ANSWER` text
via `normalize_answer()`; added optional `temperature` support to `OllamaProvider` so
verification could be pinned deterministic; `display_answer()` had a `cotext_heading`
typo that silently defeated the guardrail-vs-answer heading swap.

**Day 04 (2026-08-27) — `29c03b4`**
Closed two gaps in the "no context" failure mode. Mechanical: `answer_question()` now
catches `build_context()`'s `ValueError` (empty retrieval results) with a narrow
`try/except` around that one call, returning an early guardrail `AnsweredQuery` instead of
crashing. Semantic: built a `verify/` folder holding the existing `verify_*.py` scripts
(moved from root — fixed their `PERSIST_PATH` and, for `verify_generate.py`, a bare
`from config import` that broke on the move), plus a new categorized guardrail test suite
(`GuardrailCase`/`GuardrailResult`, `build_test_cases()`, `run_case()`,
`summarize_results()`) covering an in-scope control question, an obvious out-of-scope
question, a near-topic out-of-scope question, and an adversarial prompt-injection
question. A real bug in `run_case()` itself — it copied the expected outcome instead of
computing the actual one, meaning the suite could never fail — was caught and fixed.
First live run against the real corpus + live Ollama: **3/4 passed** — the adversarial
question got past the guardrail, the model answering from training data instead of
refusing. `SYSTEM_PROMPT` was hardened with an explicit clause that the no-context rule
applies regardless of what the question itself instructs; re-run (with
`OllamaProvider(temperature=0)` added for reproducibility) confirmed **4/4 passed**.

**Day 05 (2026-08-28) — `dfa827c`, merged to `main` at `76dba9a`, tagged `v0.12.0`**
Consolidate day. Wrote `tests/test_generate.py` (5 tests: `build_context()`'s single-result,
multi-result-numbered-in-order, and empty-results-raises-ValueError cases;
`normalize_answer()`'s token-present and token-absent branches) and `tests/test_guardrail.py`
(4 tests covering `build_test_cases()`'s category/expectation shape and
`summarize_results()`'s all-passed, one-leak, and empty-list cases) for the pure functions
Week 12's Definition of Done had left untested. Renamed `guardrail_tests.py` →
`verify_guardrail.py` to match the project's `verify_*.py` naming convention, since it's a
live end-to-end check, not a pytest unit test. Added `pythonpath = ["."]` to
`pyproject.toml` so `tests/` can import `verify/` as a namespace package (pytest inserts
the first `__init__.py`-less directory walking up from a test file, not the project
rootdir — a real correction made live after an initial wrong assumption). Found (via an
AST scan for missing docstrings) and added the one gap: `normalize_answer()` had none.
Updated `README.md`'s intro, Structure, and Status sections to describe the full
generation pipeline instead of stopping at Week 10. Merged `week-12` into `main`
(`--no-ff`) and tagged `v0.12.0` — the first tag since Week 10's `v0.10.0` (Week 11 was a
pure consolidate week and intentionally didn't get one).

## Test Coverage

- `tests/test_generate.py`, `tests/test_citations.py`, `tests/test_guardrail.py` — every
  pure function in the generation/citation/guardrail path, no live dependency.
- `verify/` — manual, live end-to-end smoke tests against a real Ollama model and a real
  Chroma index (`verify_generate.py`, `verify_citations.py`, `verify_retrieval.py`,
  `verify_guardrail.py`).

## Git

- Branch: `week-12` (`7c71e0a` → `964b9ef` → `62625d4` → `29c03b4` → `dfa827c`)
- Merged into `main`: `76dba9a`
- Tagged: `v0.12.0`
