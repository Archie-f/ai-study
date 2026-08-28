# ai-study

![CI](https://github.com/Archie-f/ai-study/actions/workflows/ci.yml/badge.svg)

## What
A week-by-week log of my AI Engineer transition: scripts, notes, and
experiments covering Python for AI, LLM APIs, RAG, agents, and deployment.

## Why
Moving from Senior Test Automation Engineer to AI Engineer. Building in public so the
learning is visible, accountable, and eventually teachable.

## How

**Requirements:** 
* Python 3.12+, 
* an Anthropic API key, 
* a Groq API key, 
* Ollama (optional)

```bash
git clone https://github.com/Archie-f/ai-study.git
cd ai-study
python -m venv .venv && source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python week-02/hello_llm.py
```

## Weekly Progress

| Week    | Topic                                                                                | Status |
|---------|----------------------------------------------------------------------------------------|--------|
| Week 01 | Python for AI, packaging, async CLI                                                    | Done   |
| Week 02 | LLM APIs, Git workflow, CI                                                              | Done   |
| Week 03 | Chatbot refactor, logging, config, pytest                                               | Done   |
| Week 04 | Prompt Engineering Foundations — few-shot, chain-of-thought, evaluation mindset         | Done   |
| Week 05 | Multi-Provider LLM Integration (llm-compare) — provider abstraction, retry/fallback     | Done   |
| Week 06 | Evaluation & Scoring — exact-match + LLM-as-judge, batch runner, regression checks      | Done   |
| Week 07 | Model Economics, Streaming, Cost Tracking & Resilience                                 | Done   |
| Week 08 | llm-compare v1.0 — installable package, CLI, README                                    | Done   |
| Week 09 | RAG: Ingestion & Chunking                                                               | Done   |
| Week 10 | RAG: Retrieval — embeddings, vector store, BM25, hybrid search                         | Done   |
| Week 11 | Consolidate #1 — rebuild from memory, Python fundamentals, framework reading            | Done   |
| Week 12 | RAG: Generation — prompt assembly, citations, guardrails, CLI                           | Done   |

## What I Learned
### Week 01
- Type hints and dataclasses — adding Java-style structure to Python without the boilerplate 
- Packaging a CLI tool with pyproject.toml — installable with pip install -e .
- async/await with httpx — non-blocking HTTP calls to a local LLM
- Building a typed CLI with Typer — flags, arguments, and --help out of the box
- Refactoring a working chatbot into a clean, installable package

### Week 02
- Managing API secrets with python-dotenv — never hardcode keys
- Building a unified ask() function that works across Claude, Groq, and Ollama
- The messages[] format shared by all modern LLM APIs
- Git rebase and interactive history cleanup
- GitHub Actions CI — automated linting (`ruff`) and tests (`pytest`) on every push

### Week 03
- Rebuilt the chatbot from scratch, typed, documented, and with proper error handling
- Added structured logging with Python's logging module — levels, handlers, latency timing
- Introduced ChatConfig dataclass — single config object replacing all magic strings
- Added tests directory, wrote tests mocking the Ollama call with unittest.mock.patch
- Defined and used pytest fixtures to share setup across tests without breaking test isolation
- Wrote a professional, scannable README file with CI and Python version badges
- Added ChatConfig class docstring, __main__.py module docstring, completed setup_logging Raises section
- Updated python-version in ci.yml from 3.11 to 3.12 to support the type alias syntax
- Added run() and setup_logging() tests - Expanded coverage to 93%
- Used "builtins.input", "builtins.print", and @pytest.mark.parametrize
- Added "pytest-cov", "mypy", and "ruff" into project.optional-dependencies
- Learned the pre-merge review workflow — mypy, pytest, ruff, README, and CI must all be green before merging to main

### Week 04
- Learned systematic prompting, few-shot prompting and chain-of-thought (CoT)
- Built an evaluation mindset: "Every prompt is a hypothesis. An eval is the experiment that tests it."
- Used Python concepts: zip(), @dataclass, textwrap.dedent(), generator expressions with sum()
- Added prompt_runner with variant experiment harness
- Structured JSON output from LLM API using prompt-based JSON mode
- Pydantic BaseModel for parsing and validating LLM responses
- Batch evaluation loop with pass rate scoring
- Failure categorisation (format failure vs content failure vs label disagreement)
- Prompt tuning loop: one change at a time, re-run batch, measure improvement
- System prompts: role framing, constraints, and format directives as engineering tools
- The Anthropic API separates system= from messages[] — other providers (OpenAI, Ollama, Gemini) handle this differently
- LLMs are stateless — conversation memory is your responsibility; you resend the full history on every call
- Built ConversationManager: stateful class managing message history with role validation (Literal type), get_history(), clear(), and send()
- Context window management with a sliding window: trim oldest user+assistant pairs to stay under token budget
- client.messages.count_tokens() counts tokens without generating output — cheap pre-call budget check

### Week 05
- Built a shared `LLMProvider` interface (`ask()`, `LLMResult`, `ProviderError`) so Claude, GPT, Groq, and Ollama all plug in the same way
- Learned Groq supports the OpenAI API format — reused the OpenAI client pointed at Groq's base URL instead of writing a separate SDK integration
- Built `retry_with_backoff()` and a `ProviderChain` fallback so one provider going down doesn't crash a comparison run
- Used dataclasses over dicts for `LLMResult` so mypy could catch shape mistakes
- Built a comparison table plus JSON save/load so provider runs are reproducible and comparable over time

### Week 06
- Built the evaluation layer: exact-match scoring for factual/sentiment categories, LLM-as-judge scoring for open-ended summarization
- Learned to degrade gracefully — a malformed judge JSON response becomes `score=None` instead of crashing the whole batch run
- Built a regression checker comparing pass-rate maps against a saved baseline, flagging any drop beyond a tolerance
- Found and fixed a real sign-inversion bug in the regression checker that was silently hiding regressions
- Learned to test LLM-integration code without live API calls — every provider mocked in the eval test suite

### Week 07
- Learned real token economics: `count_tokens()`/`estimate_cost()` from scratch, and fixed two real cost-estimation bugs (a rounding-at-scale error, a nightly-vs-monthly unit mismatch)
- Implemented streaming responses with a provider-agnostic generator consumer (`next()`/`StopIteration`)
- Built a cost/latency dashboard with CSV and Markdown report generation
- Learned to classify errors before retrying — only 429/5xx/timeout are retryable, not 400/401/403 — and gave rate-limit errors their own longer backoff
- Hit the classic Python gotcha: `None`-default parameters must be resolved inside the function body, not as literal defaults, since Python binds defaults once at definition time
- Got `week-05`/`week-07` passing `mypy --strict` cleanly

### Week 08
- Consolidated three weeks of scripts into one installable `llm_compare` package (`pip install -e .`)
- Built a real CLI (`llm-compare cost` / `llm-compare eval`) with argparse subcommands
- Learned to write a README that sells the tool's actual value (cost/latency/quality data vs. "vibes") rather than just listing files
- Shipped as `v1.0.0` — the first full 1.0 release in this project, not just a weekly script folder

### Week 09
- Built the RAG ingestion pipeline: a corpus loader plus two chunking strategies (structure-aware by heading, fixed-size with token overlap)
- Built a hand-labeled golden Q&A dataset up front, alongside the chunker, instead of deferring evaluation to a later week
- Learned to design chunker-agnostic, retrieval-mode-agnostic reference data so the same golden set works across every retrieval method tried later

### Week 10
- Wired the full retrieval pipeline: local sentence-transformers embeddings, a Chroma vector store, a from-scratch BM25 sparse index, and hybrid search via Reciprocal Rank Fusion
- Learned why hybrid retrieval matters here specifically — BM25 stays competitive out-of-domain, and personal study notes are out-of-domain text for a general embedding model
- Kept live-dependency functions (`embed_chunks()`, `get_query_result()`) deliberately out of the fast unit test suite, while still covering every pure function (`tokenize`, `score_document`, `rrf_merge`, etc.)

### Week 11 (Consolidate #1)
- Rebuilt the chunkers, embedder, and retrieval modules entirely from memory — no new pipeline code, but two real bugs caught and backported in the process
- Closed a Python-fundamentals gap list 5/5 with `mypy`, including two live-verified corrections (a false claim about `Document` as a context manager, and a false claim about `@contextmanager` guaranteeing teardown)
- Did a full reading pass over LangChain and LlamaIndex, mapping every core abstraction against this project's own design — confirmed the hand-rolled pipeline shape has direct analogues in both frameworks

### Week 12
- Built the last two stages of the RAG pipeline: prompt assembly + generation (reusing Phase 2's `LLMProvider` abstraction), then structured citations
- Built a real guardrail against hallucinated answers — a `NO_ANSWER_TOKEN` model-emitted sentinel, decoupled from the human-facing `GUARDRAIL_ANSWER` text via `normalize_answer()`
- Found and fixed a real prompt-injection vulnerability live: an adversarial question got the model to ignore its own system prompt and answer from training data; hardened `SYSTEM_PROMPT` with an explicit anti-override clause and re-verified 4/4 on the guardrail test suite
- Learned pytest's actual `sys.path` insertion rule — it adds the first `__init__.py`-less directory walking up from the test file, not the project rootdir — via a real `ModuleNotFoundError`, fixed with `pythonpath = ["."]` in `pyproject.toml`
- Shipped the end-to-end CLI (`ask.py`) tying retrieval → generation → citations → guardrail into one command
