from dataclasses import dataclass

test_cases_list: list[tuple[str, str]] = [
    ("I was charged twice", "billing"),
    ("App crashes on login", "technical"),
    ("What are your hours?", "general"),
    ("My card was declined", "billing"),       # edge case
    ("Can't reset password", "technical"),     # edge case
]

@dataclass
class Mistake:
    """Expected and actual data for an input."""
    input: str
    expected: str
    got: str

    def to_dict(self):
        """Return a dict representing the Mistake"""
        return {"input": self.input, "expected": self.expected, "got": self.got}

def evaluate_prompt(test_cases: list[tuple[str, str]], predictions: list[str]) -> dict:
    """Counts the number of correct responses and calculates the accuracy.
        Args:
            test_cases (list[tuple[str, str]]): Test cases list
            predictions (list[str]): Predictions list

        Returns:
            Returns the evaluation of the results as a dict of result
    """
    mistakes: list[Mistake] = []
    correct: int = 0
    for (inp, expected), actual in zip(test_cases, predictions):
        if actual == expected:
            correct += 1
        else:
            mistakes.append(Mistake(input=inp, expected=expected, got=actual))

    accuracy: float = correct / len(test_cases)
    return {
        "total": len(test_cases),
        "correct": correct,
        "accuracy": accuracy,
        "mistakes": [m.to_dict() for m in mistakes],
    }

if __name__ == "__main__":
    predictions_list: list[str] = ["billing", "general", "general", "technical", "technical"]
    print(evaluate_prompt(test_cases_list, predictions_list))