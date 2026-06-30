from eval_types import EvalResult

BASELINE_PASS_RATE = 0.85

def check_regression(results: list[EvalResult]) -> None:
    """Check if a regression exists in the pass rate."""
    pass_rate: float = sum(1 for r in results if r.passed) / len(results)
    diff: float = pass_rate - BASELINE_PASS_RATE

    if diff < -0.05:
        print(f"!!! REGRESSION: {diff:.0%} (delta: {diff:+.0%}), BASELINE: {BASELINE_PASS_RATE:.0%}")
    else:
        print(f"√ No Regression. Pass rate: {pass_rate:.0%} (delta: {diff:+.0%}), BASELINE: {BASELINE_PASS_RATE:.0%}")