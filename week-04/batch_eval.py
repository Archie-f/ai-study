from dataclasses import dataclass
from structured_outputs import analyse_sentiment

@dataclass
class TestData:
    """A single input/expected-output pair for batch evaluation."""
    label: str
    input_text: str
    expected_sentiment: str

@dataclass
class EvalResult:
    """Data for a test result evaluation."""
    label: str
    input_text: str
    expected_sentiment: str
    actual_sentiment: str | None
    passed: bool
    failure_reason: str | None

def run_batch_eval(test_cases: list[TestData]) -> list[EvalResult]:
    """Evaluate a batch of test cases.

        Args:
            test_cases (list[TestData]): A list of test cases.

        Returns:
            list[EvalResult]: A list of EvalResult.
    """
    eval_results: list[EvalResult] = []
    for test_case in test_cases:
        result = analyse_sentiment(test_case.input_text)
        eval_result = EvalResult(
            label=test_case.label,
            input_text=test_case.input_text,
            expected_sentiment=test_case.expected_sentiment,
            actual_sentiment=result[0].sentiment if result[0] else None,
            passed=result[0] is not None and result[0].sentiment == test_case.expected_sentiment,
            failure_reason=(
                None if result[0].sentiment == test_case.expected_sentiment
                else f"wrong sentiment: got {result[0].sentiment}, expected {test_case.expected_sentiment}"
            ) if result[0] else result[1],
        )
        eval_results.append(eval_result)
    return eval_results

def print_summary(results: list[EvalResult]) -> None:
    """Print the summary of a batch of test results.
        Args:
            results (list[EvalResult]): A list of EvalResult.
    """
    headers: list[str] = ["LABEL", "EXPECTED", "ACTUAL", "PASS/FAIL"]
    space: int = 13

    print(f"""{headers[0]:<25} {headers[1]:<{space}} {headers[2]:<{space}} {headers[3]:<{space}}""")
    for result in results:
        print(f"""{result.label:<25} {result.expected_sentiment:<{space}} {result.actual_sentiment:<{space}} {"PASSED" if result.passed else "FAILED":<{space}}""")

    total_pass: int = sum(1 for result in results if result.passed)
    total_pass_rate: float = (total_pass / len(results)) * 100
    print(f"""Total Pass Rate: {total_pass_rate} %""")

def categorize_failures(results: list[EvalResult]) -> None:
    """Categorize a batch of test results.
        Args:
            results (list[EvalResult]): A list of EvalResult.
    """
    failures: list[EvalResult] = [res for res in results if not res.passed]
    headers: list[str] = ["LABEL", "FAILURE_REASON"]

    if not failures:
        print("No failures found.")
        return

    print(f"""{headers[0]:<25} {headers[1]:<25}""")
    for fail in failures:
        print(f"""{f'[{fail.label}]':<25} {fail.failure_reason}""")

    print(f"""{len(failures)} / {len(results)} failed.""")

if __name__ == "__main__":
    test_case_list: list[TestData] = [
        TestData("sarcasm", "Oh great, another broken product. Just what I needed.", "negative"),
        TestData("very_short", "Fine.", "neutral"),
        TestData("mixed", "Great camera, terrible battery life.", "neutral"),
        TestData("emphatic_pos", "Absolutely love this. Best purchase of the year!", "positive"),
        TestData("obvious_neg", "This is completely useless and ruined my day.", "negative"),
        TestData("negation_flip", "I don't hate it, it's actually pretty decent.", "positive"),
        TestData("conditional_sentiment", "It would be perfect if it arrived on time.", "neutral"),
        TestData("slang_pos", "This device is absolutely goated!", "positive"),
    ]

    results_list: list[EvalResult] = run_batch_eval(test_case_list)
    print_summary(results_list)
    categorize_failures(results_list)
