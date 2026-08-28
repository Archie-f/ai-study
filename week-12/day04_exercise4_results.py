@dataclass
class TestResult:
    question: str
    category: str
    guardrail_expected: bool
    guardrail_fired: bool


def summarize_results(results: list[TestResult]) -> str:
    """Summarize a guardrail test run as a one-line pass/fail count.

    Args:
        results: One TestResult per test question that was run.

    Returns:
        A string like "3/4 passed, 1 unexpected leak" — a result
        counts as passed when guardrail_fired matches
        guardrail_expected.
    """
    if not results:
        return "0/0 passed, 0 unexpected leak"

    passed = sum(1 for result in results if result.guardrail_expected == result.guardrail_fired)
    failed = len(results) - passed
    leak_suffix = "leak" if failed == 1 else "leaks"

    return f"{passed}/{len(results)} passed, {failed} unexpected {leak_suffix}"
