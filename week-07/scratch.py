import os
import sys
import time
from pathlib import Path
from typing import Generator, Any

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

# Day 03 - Exercise 3 — Implement ask_stream() for One Provider
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
    prompt: list[MessageParam] = [
        {"role": "user", "content": input_text}
    ]
    response: str = ""
    start_time: float = time.perf_counter()
    with client.messages.stream(
        model=os.getenv("ANTHROPIC_MODEL_NAME"),
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
        model=os.getenv("ANTHROPIC_MODEL_NAME"),
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


if __name__ == "__main__":
    # print(calculate_estimate_cost(800, 150, 200, "claude-sonnet"))
    #
    # providers_list = ["claude-haiku", "claude-sonnet", "gpt-4o-mini"]
    # print(estimate_regression_budget(500, 300, 120, providers_list, 50))

    input_text = "Write a two-line limerick about deploying code on a Friday."
    gen = ask_stream(input_text=input_text)
    print_stream_and_collect_result(gen)
