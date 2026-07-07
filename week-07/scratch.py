# Exercise 2 — Estimate a Different Job's Cost
# A colleague wants to summarize 200 support emails, sending each through Claude Sonnet 4.6,
# expecting roughly 800 input tokens and 150 output tokens per email
# (different numbers and a different scenario from the code above — do not reuse the eval-harness example).
# Using PROVIDER_PRICING and estimate_cost(), calculate:
#   (a) the cost of a single email, and
#   (b) the total cost for all 200 emails if run one at a time.

# Expected output format:
#   two float values (or a tuple of two floats) — cost per email, and total cost for the batch — each rounded to 6 decimal places.

from token_utils import estimate_cost, estimate_batch_cost

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


# Exercise 3 — Budget a Regression Suite
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



if __name__ == "__main__":
    print(calculate_estimate_cost(800, 150, 200, "claude-sonnet"))

    providers_list = ["claude-haiku", "claude-sonnet", "gpt-4o-mini"]
    print(estimate_regression_budget(500, 300, 120, providers_list, 50))