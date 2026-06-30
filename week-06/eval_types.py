from dataclasses import dataclass

@dataclass
class EvalCase:
    """One row in your eval dataset."""
    prompt: str
    expected: str
    category: str = ''
    task_description: str = ''

@dataclass
class EvalResult:
    """The outcome of scoring one case."""
    case: EvalCase
    actual_output: str
    score: float
    passed: bool
    reason: str = ''
