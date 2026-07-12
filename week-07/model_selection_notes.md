# Model Selection Notes — llm-compare

Real numbers from a 5-prompt batch (factual, summarization, code generation, reasoning, instruction-following) run across all four providers via `run_comparison_batch()`, judged by GPT-4o-mini on a 0–1 quality scale.

| Provider | Calls | Total Cost | Avg Latency | Quality |
|---|---|---|---|---|
| groq (llama-3.1-8b-instant) | 5 | $0.0000 | 357 ms | 0.9 |
| claude (claude-haiku-4-5) | 5 | $0.0038 | 2766 ms | 0.8 |
| open_ai (gpt-4o-mini) | 5 | $0.0004 | 2486 ms | 0.7 |
| ollama (llama3, local) | 5 | $0.0000 | 24200 ms | 0.7 |

## When to pick each

- **Groq** is the default choice for this project: effectively free, by far the fastest (~0.36s), and the highest quality score in this batch — hard to beat on any axis.
- **Claude Haiku** is worth it when a task needs the most careful reasoning or writing quality and the ~8x cost over GPT-4o-mini is trivial in absolute terms (still under half a cent per call).
- **GPT-4o-mini** is a reasonable fallback when Groq is unavailable — cheap and mid-pack on speed and quality, but didn't stand out on any single axis in this batch.
- **Ollama** only makes sense when cost must be exactly $0 regardless of time, or the data can't leave the machine — a 24-second average response is too slow for anything interactive.

## When local beats hosted

Ollama wins specifically when working offline, handling sensitive data that shouldn't leave the machine, or running high-volume batch jobs where a 24-second-per-call latency doesn't matter and $0 marginal cost does. For this project's day-to-day use — fast iteration while developing `llm-compare` itself — a hosted API (Groq, in this batch) is the better default.
