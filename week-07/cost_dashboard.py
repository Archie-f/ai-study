import csv
import dataclasses
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / 'week-05'))
from provider import LLMResult, LLMProvider


LOG_PATH: Path = Path(__file__).parent / "results" / "cost_log.jsonl"

def log_run(result: LLMResult, log_path: Path = LOG_PATH) -> None:
    """Append one LLMResult as a single JSON line to the cost log."""
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


def summarize(log_path: Path = LOG_PATH) -> dict[str, dict]:
    """Aggregate a cost log into per-provider totals and averages.

    Returns:
        {provider: {"calls": int, "total_cost": float, "avg_latency_ms": float}}
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    with log_path.open() as f:
        for line in f:
            entry = json.loads(line)
            grouped[entry["provider"]].append(entry)

    summary: dict[str, dict] = {}
    for provider, entries in grouped.items():
        calls = len(entries)
        total_cost = round(sum(entry["cost"] for entry in entries), 6)
        avg_latency_ms = round((sum(entry["latency_ms"] for entry in entries) / calls), 1)
        summary[provider] = {
            "calls": calls,
            "total_cost": total_cost,
            "avg_latency_ms": avg_latency_ms,
        }

    return summary


def write_cost_summary_csv(summary: dict[str, dict], out_path: Path) -> None:
    """Write summarize()'s output as a CSV file.

    Columns: provider, calls, total_cost, avg_latency_ms — one row per
    provider in summary. out_path is the destination CSV file itself.
    """
    headers = ["provider", "calls", "total_cost", "avg_latency_ms"]
    rows = []
    for provider in summary.keys():
        stats = summary[provider]
        rows.append([
            provider,
            stats["calls"],
            stats["total_cost"],
            stats["avg_latency_ms"]
        ])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)



if __name__ == "__main__":
    import sys

    sys.path.insert(0, '../week-05')
    from dotenv import load_dotenv

    load_dotenv('../.env')

    from anthropic_provider import AnthropicProvider


    provider = AnthropicProvider()
    result = tracked_call(provider, "Reply with exactly one word: hello")
    print(result)

    print(summarize(LOG_PATH))