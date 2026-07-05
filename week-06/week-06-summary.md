# Week 06 Summary — Evaluation & Scoring

Week 06 adds the evaluation layer to `llm-compare`: the part that automatically judges which provider gave the best answer to a prompt. Days 01–04 built the layer piece by piece; this file documents the finished state as of Day 05 (v0.6.0).

## Files

**`eval_types.py`** — `EvalCase` (prompt, expected answer, category, task description) and `EvalResult` (case, actual output, score, passed, reason). The two dataclasses every other file in the package passes around.

**`exact_match.py`** — `normalize()` strips noise symbols and lowercases; `score_exact()` does case-insensitive, punctuation-insensitive string comparison. Used for categories with one unambiguous correct answer (factual, sentiment).

**`llm_judge.py`** — LLM-as-judge scoring for open-ended categories (summarization) where there's no single correct string. `build_judge_prompt()` frames the task for the judge model; `score_with_llm()` sends it to a judge provider, asks for a 0–3 score plus reason as JSON, and normalizes the score to 0.0–1.0. `clean_result()` strips markdown code fences the judge sometimes wraps its JSON in. A malformed judge response degrades to `score=None` rather than crashing the run.

**`eval_harness.py`** — `SCORER_DISPATCH` maps category → scorer function (`factual`/`sentiment` → `score_exact`, `summarization` → `score_with_llm`). `run_eval()` runs a list of cases through one provider, dispatching each case to its scorer. `print_report()` prints a pass/fail summary to the terminal.

**`regression_check.py`** — `load_baseline()` reads a `dict[str, float]` of previously recorded pass rates from JSON. `check_regression()` compares a baseline against a current rate map and flags any key that dropped by more than a tolerance (default 5%). Deliberately takes two already-computed rate maps rather than raw results, so the same function works for both provider-level and category-level comparisons.

**`eval_dataset.py`** — `EVAL_DATASET`: 10 `EvalCase` objects across three categories (factual, summarization, sentiment) via the `Category` enum. The real dataset used by the batch runner and the report.

**`batch_runner.py`** — `estimate_cost()` gives a rough USD estimate before spending money on a live run, using per-provider token pricing. `run_batch()` runs the full dataset against every configured provider, optionally scored by a judge, and persists results to `results/batch_YYYY-MM-DD_HH-MM.json` via `dataclasses.asdict()`.

**`compare.py`** (in `week-05/`, extended in Week 06 Day 03) — `run_comparison()` now accepts an optional `judge` parameter; when present, each `LLMResult` gets a `judge_score`/`judge_reason` via `score_with_llm()`, with judge failures caught and turned into `score=None` rather than an exception. `print_table()` shows the Judge Score column only when at least one result has one.

**`scored_compare.py`** — End-to-end demo: one prompt, four providers, one judge, printed as a scored comparison table.

**`eval_report.py`** — `load_results()` reconstructs `EvalResult`/`EvalCase` objects from a saved batch JSON file. `get_pass_rates_per_provider()` / `get_pass_rates_per_category()` turn raw results into `dict[str, float]` rate maps (the same shape `regression_check.py` expects). `generate_report()` prints an overall pass rate, per-provider and per-category tables, and failure-only detail (passing cases aren't printed individually — noise reduction). The CLI entry point supports `--check-regression BASELINE_PATH` to run a regression check against a saved baseline.

**`run_mini_eval.py`** — Day 01 learning script; still useful as a small, fast end-to-end example separate from the full batch/report pipeline.

**`tests/`** — `test_batch_runner.py` (scorer dispatch, persistence, cost estimation error handling), `test_scored_compare.py` (judge integration in `compare.py`: success, wrong answer, malformed judge JSON, judge call failure), `test_eval_report.py` (rate calculations, regression detection including the "category with zero cases" and "key missing from current" edge cases). All providers are mocked — no real API calls in the test suite.

## How this connects to llm-compare

Before Week 06, `llm-compare` could run a prompt across providers and show cost/latency (Week 05). It could not tell you which answer was actually *better*. The evaluation layer closes that gap: `batch_runner.py` + `eval_report.py` give a repeatable way to run a fixed set of test cases against every provider and get back a scored, categorized report — the feature that turns the tool from "a multi-provider runner" into "a tool that helps you decide which provider to use for which kind of task." This is the core of the "evaluation story" the README highlights, and the direct foundation for Week 07's cost/latency observability layer and Week 08's v1.0.0 consolidation.

## Known non-blocking finding

`regression_check.py` from Day 04 has a real bug fixed during that session (sign inversion that hid regressions) — already corrected and covered by `test_check_regression_flags_drop_and_skips_missing_key`. No open issues carried into Day 05.

## Verification status

Automated `pytest` and `mypy` were not run inside this session (sandbox has no network access to install dependencies and the project's `.venv` is macOS-specific). Akif ran both locally before merge — see the Day 05 takeaway notes for the actual output.
