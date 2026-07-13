# Week 07 Summary — Model Economics, Streaming, Cost Tracking & Resilience

## What Was Built

- **Day 01 — Token Economics & Pricing:** `token_utils.py` — `PROVIDER_PRICING`, `count_tokens()`, `estimate_cost()`, `estimate_batch_cost()`. Fixed two real cost-estimation bugs (rounding-then-scaling error at batch scale; a nightly-vs-monthly budget unit mismatch).
- **Day 02 — Streaming Responses:** `provider.py` gained `ask_stream()`; implemented for `AnthropicProvider`. `streaming_demo.py` — a provider-agnostic stream consumer using the generator `next()`/`StopIteration` pattern.
- **Day 03 — Cost & Latency Dashboard:** `cost_dashboard.py` — `log_run()`, `tracked_call()`, `summarize()` (per-provider calls/cost/latency/quality), plus CSV and Markdown report generators. Also closed a gap left over from Week 06: `eval_harness.py` now keeps the full `LLMResult` instead of discarding it, and `eval_report.py` persists `results/report_<timestamp>.md` instead of only printing.
- **Day 04 — Model Selection & Resilience:** `retry_backoff.py` — `retry_with_backoff()` with exponential backoff, jitter, and a longer dedicated delay for rate-limit errors. Wired into both `compare.py` and `eval_harness.py`, so a single bad API call no longer crashes a batch. `model_selection_notes.md` — a real cost/latency/quality decision guide from an actual 4-provider batch run.
- **Day 05 — Consolidate & Ship:** Flattened `week-07/` to match `week-05/`/`week-06/`'s layout; fixed remaining bugs in the cost report generators; `week-07/` and `week-05/` now pass `mypy --strict` cleanly (fixed missing type annotations, bare `dict` generics, and a `Generator`/`Any` re-export gap in the provider modules); `requirements.txt` cleaned up (dropped a duplicate `groq` line and two unused packages, added version floors); added a `ruff.toml` documenting two deliberate repo-wide style choices; this file.

## Key Decisions

1. **Retry classification over blanket retries.** Only 429/5xx/timeout errors get retried; 400/401/403 fail immediately (`ProviderError.retryable`). Retrying a permanent failure just burns time and API cost for no benefit.
2. **Rate limits get their own backoff.** A 429 means the provider is certain you're going too fast, not unlucky — `retry_with_backoff()` gives `RateLimit` errors a 5x longer base delay than other retryable errors by default.
3. **`log_run()` moved to after judge scoring in `run_comparison()`.** Logging before scoring meant `judge_score` was always null in the cost log — cost, latency, and quality now come from the same generation.
4. **Quality averages skip nulls, not just divide by call count.** Dividing by total calls silently understated quality whenever any judge call failed for that provider.
5. **`None`-default path parameters resolved inside the function body**, not as literal defaults — a recurring bug this week (`LOG_PATH`, `CSV_PATH`, a `since` timestamp) since Python binds default argument values once, at function definition time.

## Deferred / Known Gaps

- **CI is red on `main`.** `ruff check .` still fails on 33 pre-existing `F403`/`F405` errors caused by `from provider import *` in `week-05`'s three provider modules (`anthropic_provider.py`, `openai_provider.py`, `ollama_provider.py`) — predates this week, not something introduced here. Fix is a real one: convert each to explicit imports. Deferred to a future week.
- A handful of smaller pre-existing `ruff` findings outside this week's scope (an unused import in `compare.py`, one in `week-06/tests/test_scored_compare.py`, etc.) — untouched.
- `mypy --strict` and `ruff check .` were resolved for everything shipped this week (`week-07/` is fully clean under both); `scratch.py` is intentionally excluded from the `mypy` gate (exploratory/practice code, not a shipped module) via `mypy.ini`.
