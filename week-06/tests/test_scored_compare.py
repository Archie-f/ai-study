import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'week-05'))
from compare import ComparisonResult, run_comparison, print_table


prompt: str = "Explain what is API in one sentence."

def test_correct_llm_answer():
    mock_judge = MagicMock()
    mock_judge.ask.return_value = MagicMock(text='{"score": 3, "reason": "The answer is accurate."}')

    mock_provider = MagicMock()
    mock_provider.ask.return_value = MagicMock(text="API is Application Programming Interface.")

    comparison_result = run_comparison(prompt, [mock_provider], judge=mock_judge)
    for r in comparison_result.results:
        assert r.judge_score == 1
        assert  r.judge_reason == "The answer is accurate."

def test_wrong_llm_answer():
    mock_judge = MagicMock()
    mock_judge.ask.return_value = MagicMock(text='{"score": 0, "reason": "Invalid answer."}')

    mock_provider = MagicMock()
    mock_provider.ask.return_value = MagicMock(text="API is Active Pharmaceutical Ingredient.")

    comparison_result = run_comparison(prompt, [mock_provider], judge=mock_judge)
    for r in comparison_result.results:
        assert r.judge_score == 0
        assert  r.judge_reason == "Invalid answer."

def test_failed_llm_answer():
    mock_provider = MagicMock()
    mock_provider.ask.return_value = MagicMock(text="Error occurred.")

    mock_judge = MagicMock()
    mock_judge.ask.return_value = MagicMock(text="Invalid JSON.")

    comparison_result = run_comparison(prompt, [mock_provider], judge=mock_judge)
    for r in comparison_result.results:
        assert r.judge_score is None
        assert r.judge_reason.startswith("Judge returned invalid JSON:")

def test_judge_call_fails():
    mock_provider = MagicMock()
    mock_provider.ask.return_value = MagicMock(text="API is Application Programming Interface.")

    mock_judge = MagicMock()
    mock_judge.ask.side_effect = RuntimeError("Judge returned timeout error.")

    comparison_result = run_comparison(prompt, [mock_provider], judge=mock_judge)
    for r in comparison_result.results:
        assert r.judge_score is None
        assert r.judge_reason == "Error: Judge returned timeout error."