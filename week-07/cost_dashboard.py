import csv
import dataclasses
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent / 'week-05'))
from provider import LLMResult, LLMProvider
from openai_provider import OpenAIProvider


LOG_PATH: Path = Path(__file__).parent / "results" / "cost_log.jsonl"
CSV_PATH: Path = Path(__file__).parent / "results" / "cost_summary.csv"
RESULTS_PATH: Path = Path(__file__).parent / "results"

def log_run(result: LLMResult, log_path: Path | None = None) -> None:
    """Append one LLMResult as a single JSON line to the cost log."""
    if log_path is None: log_path = LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = dataclasses.asdict(result)
    entry["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with log_path.open(mode="a") as f:
        f.write(json.dumps(entry) + "\n")

def tracked_call(
    provider: LLMProvider, user_input: str, system_prompt: str = ""
) -> LLMResult:
    """Call provider.ask() and log the resulting metrics before returning it."""
    result = provider.ask(user_input=user_input, system_prompt=system_prompt)
    log_run(result)
    return result


def summarize(log_path: Path | None = None, since: str | None = None) -> dict[str, dict[str, Any]]:
    """Aggregate a cost log into per-provider totals and averages.

    Returns:
        {provider: {"calls": int, "total_cost": float, "avg_latency_ms": float}}
    """
    if log_path is None: log_path = LOG_PATH
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with log_path.open() as f:
        for line in f:
            entry = json.loads(line)
            if since is not None and  entry["timestamp"] < since:
                continue
            grouped[entry["provider"]].append(entry)

    summary: dict[str, dict[str, Any]] = {}
    for provider, entries in grouped.items():
        calls = len(entries)
        total_cost = round(sum(entry["cost"] for entry in entries), 6)
        avg_latency_ms = round((sum(entry["latency_ms"] for entry in entries) / calls), 1)
        scored = [entry["judge_score"] for entry in entries if entry["judge_score"] is not None]
        quality = round(sum(scored) / len(scored), 1) if scored else None
        summary[provider] = {
            "calls": calls,
            "total_cost": total_cost,
            "avg_latency_ms": avg_latency_ms,
            "quality": quality
        }

    return summary


def generate_cost_summary_csv(summary: dict[str, dict[str, Any]], out_path: Path | None = None) -> None:
    """Write summarize()'s output as a CSV file.

    Columns: provider, calls, total_cost, avg_latency_ms — one row per
    provider in summary. out_path is the destination CSV file itself.
    """
    if out_path is None: out_path = RESULTS_PATH

    headers = ["provider", "calls", "total_cost", "avg_latency_ms", "quality"]
    rows = []
    for provider in summary.keys():
        stats = summary[provider]
        rows.append([
            provider,
            stats['calls'],
            stats['total_cost'],
            stats['avg_latency_ms'],
            stats['quality']
        ])

    out_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = out_path / f"cost_summary_table_{timestamp}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def generate_cost_summary_markdown(summary: dict[str, dict[str, Any]], out_path: Path | None = None) -> Path:
    if out_path is None: out_path = RESULTS_PATH

    lines = [f"# Summary Report", "\n## Number of calls, Cost, Latency and Quality — Per Provider\n",
             "| Provider | Calls | Total Cost (USD) | Avg Latency (ms) | Quality |", "|---|---|---|---|---|"]

    for provider in summary.keys():
        stats = summary[provider]
        lines.append(f"| {provider} | {stats['calls']} | {stats['total_cost']} | {stats['avg_latency_ms']} | {stats['quality']} |")

    report_text = "\n".join(lines)

    out_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = out_path / f"cost_summary_report_{timestamp}.md"
    path.write_text(report_text)
    print(f"Report saved to {path}")

    return path

def run_comparison_batch(prompts: list[str], providers: list[LLMProvider]) -> None:
    """Run a batch of prompts through run_comparison(), then report cost,
    latency, and quality for just this batch.

    Each prompt is run across all given providers via run_comparison(),
    scored by an OpenAI judge, and logged to the cost log as it goes
    (run_comparison() already calls log_run() internally, after judging).
    Once the whole batch is done, summarize() is scoped to only this run's
    entries via `since`, so older log data (e.g. from a previous session)
    isn't mixed into the results. The combined per-provider summary
    (calls, total_cost, avg_latency_ms, quality) is printed and also
    written out as a CSV via generate_cost_summary_csv().

    Args:
        prompts: The prompts to run through every provider.
        providers: The LLMProvider instances to compare.
    """
    from compare import run_comparison

    since = datetime.now().isoformat(timespec="seconds")
    judge = OpenAIProvider()
    for prompt in prompts:
        run_comparison(prompt, providers, judge=judge)

    summary = summarize(since=since)
    print(f"Summary: {summary}")
    generate_cost_summary_csv(summary)
    generate_cost_summary_markdown(summary)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, '../week-05')
    from dotenv import load_dotenv

    load_dotenv('../.env')

    # from anthropic_provider import AnthropicProvider
    #
    #
    # provider = AnthropicProvider()
    # result = tracked_call(provider, "Reply with exactly one word: hello")
    # print(result)

    print(summarize(LOG_PATH))