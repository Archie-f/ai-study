import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Generator, Any, Callable

import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv

from token_utils import estimate_cost, estimate_batch_cost

# Day 01 - Exercise 2 — Estimate a Different Job's Cost
# A colleague wants to summarize 200 support emails, sending each through Claude Sonnet 4.6,
# expecting roughly 800 input tokens and 150 output tokens per email
# (different numbers and a different scenario from the code above — do not reuse the eval-harness example).
# Using PROVIDER_PRICING and estimate_cost(), calculate:
#   (a) the cost of a single email, and
#   (b) the total cost for all 200 emails if run one at a time.
# Expected output format:
#   two float values (or a tuple of two floats) — cost per email, and total cost for the batch — each rounded to 6 decimal places.


def calculate_estimate_cost(
        input_tokens: int,
        output_tokens: int,
        number_of_emails: int,
        provider: str
) -> tuple[float, float]:
    estimate_per_email = estimate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider=provider,
    )
    estimate_for_total_emails = estimate_cost(
        input_tokens=input_tokens * number_of_emails,
        output_tokens=output_tokens * number_of_emails,
        provider=provider,
    )
    return estimate_per_email, estimate_for_total_emails


# Day 01 - Exercise 3 — Budget a Regression Suite
# Your team wants to expand Week 06's eval dataset into a 500-case regression suite,
# run nightly against Claude Haiku, Claude Sonnet, and GPT-4o-mini
# (a larger scale and different provider mix than any example in these notes).
# Assume an average of 300 input tokens and 120 output tokens per case.
# Using estimate_batch_cost(), calculate the estimated nightly cost for each provider,
# and identify which provider you would drop first if the monthly budget for this suite is capped at $50.

# Expected output format: a dict[str, float] of provider -> nightly cost, plus one sentence naming the provider to drop and why.

def estimate_regression_budget(
        number_of_cases: int,
        input_tokens: int,
        output_tokens: int,
        providers: list[str],
        budget: float,
) -> tuple[dict[str, float], str]:
    estimate_cost_per_provider = estimate_batch_cost(
        num_cases=number_of_cases,
        avg_input_tokens=input_tokens,
        avg_output_tokens=output_tokens,
        providers=providers,
    )
    total_monthly_cost = sum(estimate_cost_per_provider.values()) * 30

    if budget < total_monthly_cost:
        max_cost_provider, max_cost = max(estimate_cost_per_provider.items(), key=lambda s: s[1], default=(None, 0.0))
        conclusion = (f"The provider '{max_cost_provider}' should be dropped, because its nightly cost of '{max_cost}$' "
                      f"makes the total cost '{total_monthly_cost}$' exceed the monthly budget '{budget}$'")
    else:
        conclusion = f"The budget '{budget}$' is covering all costs '{total_monthly_cost}$'. There's no need to drop any provider."

    return estimate_cost_per_provider, conclusion


########################################################################################################################

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent / 'week-05'))
from provider import LLMResult

# Day 02 - Exercise 3 — Implement ask_stream() for One Provider
# Implement the ask_stream() signature above for AnthropicProvider, using either the generator return-value trick
# from 3.1 or the mutable-object alternative from the tip above — your choice.
# Expected output format: a script (e.g. streaming_demo.py) that calls ask_stream()
# on an AnthropicProvider instance, prints each chunk live as it arrives, and after the stream ends,
# prints an LLMResult-equivalent (text, tokens_in, tokens_out, latency_ms) built from whichever mechanism you chose.

def ask_stream(
        input_text: str,
        system_prompt: str = ''
) -> Generator[str, Any, LLMResult]:
    client = anthropic.Anthropic()
    # resolved once, with a real fallback, instead of passing os.getenv(...)
    # (str | None) straight into calls that expect a str — same bug already
    # fixed for real in week-05/anthropic_provider.py.
    model_name: str = os.getenv("ANTHROPIC_MODEL_NAME") or "claude-sonnet-4-6"
    prompt: list[MessageParam] = [
        {"role": "user", "content": input_text}
    ]
    response: str = ""
    start_time: float = time.perf_counter()
    with client.messages.stream(
        model=model_name,
        max_tokens=256,
        system=system_prompt,
        messages=prompt,
    ) as stream:
        for text in stream.text_stream:
            response += text
            yield text
        final_message = stream.get_final_message()
    elapsed_time = (time.perf_counter() - start_time) * 1000
    return LLMResult(
        provider="claude",
        model=model_name,
        text=response,
        tokens_in=final_message.usage.input_tokens,
        tokens_out=final_message.usage.output_tokens,
        latency_ms=round(elapsed_time),
    )

def print_stream_and_collect_result(generator) -> LLMResult:
    try:
        while True:
            print(next(generator), end="", flush=True)
    except StopIteration as end:
        print("\n----------------------------------------")
        print(f" * Response: \n{end.value.text}", flush=True)
        print(f" * Tokens In: {end.value.tokens_in}", flush=True)
        print(f" * Tokens Out: {end.value.tokens_out}", flush=True)
        print(f" * Elapsed Time: {end.value.latency_ms}", flush=True)
        return end.value


########################################################################################################################

# Day 03 - Exercise 1 — A Request Log for a Web Server
# A teammate wants every incoming HTTP request logged as one line, so a log-shipping tool (like Filebeat)
# can tail the file live and forward new lines as they're written. Each entry needs the
# request method, path, response status, how long it took, and a timestamp generated inside the function.

def append_json_line(entry: dict, log_path: Path) -> None:
    """Append one dict as a single JSON line to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def append_request_log(
        method: str, path: str, status: int, duration_ms: float, log_path: Path
) -> None:
    """Append one HTTP request event as a single JSON line to log_path.

    Each line must be a complete, independently-parseable JSON object
    containing method, path, status, duration_ms, and an ISO 8601
    timestamp generated inside the function.
    """
    entry: dict[str, Any] = {
        "method": method,
        "path": path,
        "status": status,
        "duration_ms": duration_ms,
        "timestamp": datetime.now().isoformat(),
    }
    append_json_line(entry, log_path)

# Day 03 - Exercise 2 — Tracking Slow File Reads
# A data pipeline reads dozens of files, and you want to know which ones are slow to read —
# without changing every call site that reads a file.
# Write a wrapper that reads a file's text, times it, and logs the result,
# then still returns the text like a normal read would.

def tracked_read(
        path: Path, log_path: Path = Path("results/read_log.jsonl")
) -> str:
    """Read path as text, measure elapsed time with time.perf_counter(),
    and append one JSON line to log_path containing
    {'file': str(path), 'chars': <int>, 'elapsed_ms': <float>}.

    Returns the file's text content, unchanged.
    """
    start_time = time.perf_counter()
    text = path.read_text()
    elapsed_time = (time.perf_counter() - start_time) * 1000

    entry: dict[str, Any] = {
        "file": str(path),
        "chars": len(text),
        "elapsed_ms": elapsed_time,
    }
    append_json_line(entry, log_path)
    return text

# Day 03 - Exercise 3 — Summarizing Page Views by Path
# An analytics pipeline writes one JSON line per page view: {"path": str, "duration_ms": float}.
# Write a function that groups these by path and reports how many views each page got and the average time spent.

def summarize_by_path(log_path: Path) -> dict[str, dict]:
    """Aggregate page view events by path.

    Returns:
        {path: {"views": int, "avg_duration_ms": float}}
    """
    grouped: dict[str, list[dict]] = defaultdict(list)

    with log_path.open() as f:
        for line in f:
            entry: dict[str, Any] = json.loads(line)
            grouped[entry["path"]].append(entry)

    summary: dict[str, dict] = {}
    for path, entries in grouped.items():
        views = len(entries)
        avg_duration_ms = round((sum(entry["duration_ms"] for entry in entries) / views), 1)
        summary[path] = {
            "views": views,
            "avg_duration_ms": avg_duration_ms,
        }

    return summary

# Day 03 - Exercise 4 — Exporting the Cost Dashboard as CSV
# summarize()'s dict output is great for code, but a teammate wants to open the numbers in a spreadsheet.
# Write a function that persists it as a CSV file — a different output format than the Markdown report you just built,
# but the same “build content, then write once” shape.
#
# write_cost_summary_csv() passed review and was promoted into week-07/cost_dashboard.py,
# right after summarize() — it's a real project feature now (a spreadsheet-friendly view of the cost dashboard),
# not just exercise practice. Import it from there if needed.

########################################################################################################################

# Day 04 - Exercise 1 — Retry Classifier for a File Downloader
# You're writing a small utility that downloads files over HTTP and needs to decide whether a failed download
# is worth retrying. It is not the LLM code from this section — it's a plain HTTP status classifier.
#
# Write it in scratch.py, then check it against a handful of codes by hand: 429, 500, 503, 400, 404, 401.

def is_retryable(status_code: int) -> bool:
    """Return True if a failed download at this status code is worth retrying.

    Treat 429 and any 5xx as retryable. Treat anything else
    (400s other than 429, or an unrecognized code) as not.
    """
    return status_code == 429 or 499 < status_code < 600


# Day 04 - Exercise 2 — Retry Loop for a Queue Consumer
# A message-queue consumer polls for the next message and occasionally gets a transient connection error
# from the queue client. Different domain from the LLM retry, same underlying pattern.
# Test it with a fake poll_fn that fails twice then succeeds, and one that always fails.

def poll_with_retry(poll_fn, max_retries=3, base_delay=0.5):
    """Call poll_fn() (which may raise ConnectionError), retrying
    with exponential backoff. Re-raise after max_retries failures.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return poll_fn()
        except ConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                jitter = random.uniform(0, 0.5)
                delay = base_delay * (2 ** attempt) + jitter
                time.sleep(delay)

    assert last_error is not None
    raise last_error

# Day 04 - Exercise 3 — Two Backoff Policies
# A print shop's job queue has two failure modes from its printer driver:
# PrinterBusyError (retry fast, printer frees up quickly) and
# PrinterOutOfPaperError (retry slow — someone has to physically refill it).
# The point of this exercise is deciding the branch, not the sleep math you already wrote in Exercise 2.

def retry_print_job(fn, is_out_of_paper: Callable[[Exception], bool]):
    """Retry fn(). Use a 5s base delay if is_out_of_paper(err) is
    True, otherwise a 0.5s base delay. Both cases still cap at
    3 attempts total.
    """
    last_error = None
    max_attempts = 3

    for attempt in range(max_attempts):
        base_delay = 0.5
        try:
            return fn()
        except Exception as e:
            last_error = e
            if is_out_of_paper(e):
                base_delay = 5
            if attempt < max_attempts - 1:
                jitter = random.uniform(0, 0.5)
                delay = base_delay * (2 ** attempt) + jitter
                time.sleep(delay)
    assert last_error is not None
    raise last_error

# Day 04 - Exercise 4 — Write model_selection_notes.md
# This code is generated to generate real results for model_selection_notes
from cost_dashboard import run_comparison_batch

sys.path.append(str(Path(__file__).resolve().parent / 'week-05'))
from provider import LLMProvider
from anthropic_provider import AnthropicProvider
from groq_provider import GroqProvider
from ollama_provider import OllamaProvider
from openai_provider import OpenAIProvider

PROMPTS: list[str] = [
    "What year did the Berlin Wall fall, and who was the German chancellor at the time?",
    "Summarize this in two sentences: [paste a paragraph from any of your project's docstrings or README]",
    "Write a Python function that checks if a string is a palindrome, ignoring case and spaces.",
    "A train leaves at 2pm going 60 mph. Another leaves the same station at 3pm going 90 mph in the same direction. At what time does the second train catch up?",
    "List exactly 3 pros and 3 cons of using a local LLM instead of a hosted API, as a numbered list, nothing else."
    ]

providers_list: list[LLMProvider] = [
    AnthropicProvider(),
    OpenAIProvider(),
    GroqProvider(),
    OllamaProvider(),
]

if __name__ == "__main__":
    # print(calculate_estimate_cost(800, 150, 200, "claude-sonnet"))
    #
    # providers_list = ["claude-haiku", "claude-sonnet", "gpt-4o-mini"]
    # print(estimate_regression_budget(500, 300, 120, providers_list, 50))

    # input_text = "Write a two-line limerick about deploying code on a Friday."
    # gen = ask_stream(input_text=input_text)
    # print_stream_and_collect_result(gen)

    # path_to_log: Path = Path(__file__).parent / "exercise_data" / "analytics.jsonl"
    # summarized = summarize_by_path(path_to_log)
    # for path in summarized.keys():
    #     print(f"Path: {path}")
    #     print(f"Views: {summarized[path]['views']}")
    #     print(f"Average duration: {summarized[path]['avg_duration_ms']}")
    #     print("-" * 33)

    # write_cost_summary_csv() now lives in cost_dashboard.py — see the demo there.
    # path_to_cost_log = Path(__file__).parent / "results" / "cost_log.jsonl"
    # path_for_cost_summary = Path(__file__).parent / "results" / "cost_summary.csv"
    # write_cost_summary_csv(
    #     summarize(path_to_cost_log),
    #     path_for_cost_summary
    # )

    run_comparison_batch(prompts=PROMPTS, providers=providers_list)



