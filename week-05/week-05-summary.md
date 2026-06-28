# Week 05 — Multi-Provider LLM Integration

## What I Built

This week I built a system that can send the same prompt to four different LLM providers — Claude, GPT, Groq, and Ollama — and get back results in the same format. I created a shared interface so each provider works the same way from the outside. I also built a comparison table to show results side by side, saved results to JSON files, and added retry and fallback logic for when a provider fails.

## File Guide

- `provider.py` — defines `LLMProvider` (the base class all providers extend), `LLMResult` (the shared return type), and `ProviderError` (custom exception with a `retryable` flag).
- `anthropic_provider.py` — implements `ask()` for Claude using the Anthropic SDK.
- `openai_provider.py` — implements `ask()` for GPT using the OpenAI SDK.
- `groq_provider.py` — implements `ask()` for Groq using the OpenAI-compatible client pointed at Groq's API.
- `ollama_provider.py` — implements `ask()` for Ollama using direct HTTP requests (no SDK).
- `openai_client.py` — raw OpenAI API call, written on Day 1 to learn the SDK before building the provider.
- `groq_client.py` — raw Groq API call, written on Day 2 to learn the API before building the provider.
- `ollama_client.py` — raw Ollama REST call, written on Day 2 for the same reason.
- `run_all_providers.py` — runs the same prompt through all four providers and prints the raw results. Used for manual testing.
- `run_side_by_side.py` — runs the same prompt through Claude, GPT, and Groq and prints a short summary for each.
- `compare.py` — holds `ComparisonResult`, `run_comparison()` to call all providers, `print_table()` to show results in a table, and `save_to_json()` / `load_from_json()` to save and load results.
- `robust_client.py` — `with_retry()` decorator that retries a failed call up to 3 times with exponential backoff, and `ProviderChain` that tries the next provider if the current one fails.
- `tests/test_compare.py` — unit tests for `run_comparison()`, `best_cost()`, `fastest()`, JSON save/load, and the empty results case.
- `tests/test_retry.py` — unit tests for retry success, non-retryable fast-fail, retry exhaustion, and `ProviderChain` fallback.
- `tests/conftest.py` — adds `week-05/` to `sys.path` so test imports work correctly.
- `results/` — JSON output files from past comparison runs, one file per run.
- `requirements.txt` — lists all external packages needed: `anthropic`, `openai`, `groq`, `requests`, `python-dotenv`, `pytest`.
- `.env.example` — shows which environment variables are needed: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `GROQ_BASE_URL`.

## How It Connects to llm-compare

The provider abstraction built this week is the core of the llm-compare project. `LLMResult` is the shared data format that makes it possible to compare providers. The four provider classes are the backends the CLI will use. `compare.py` is the comparison logic. `robust_client.py` makes the tool reliable when a provider is down.

## How to Run

```bash
# 1. Set environment variables
cp .env.example .env
# fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, GROQ_BASE_URL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a comparison across all four providers
cd week-05/
python compare.py

# 4. Run all providers individually
python run_all_providers.py

# 5. Run tests
python -m pytest tests/ -v
```

## Key Decisions

**`LLMResult` as a dataclass instead of a dict** — a dict would work but has no type checking and no methods. Using a dataclass makes the code cleaner and lets mypy catch mistakes.

**OpenAI client for Groq** — Groq supports the OpenAI API format, so I used `openai.OpenAI` with Groq's base URL instead of the Groq SDK. This keeps the code simpler and shows the OpenAI client can work with other providers too.

**`ProviderChain` as a separate class** — I could have put fallback logic inside `run_comparison()`, but that would mix two different things. `ProviderChain` implements the same `ask()` interface as a regular provider, so it can be used anywhere a provider is expected.
