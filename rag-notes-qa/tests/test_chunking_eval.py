from rag_notes.models import RetrievalEvalReport
from verify.verify_chunking_eval import format_comparison_table


def test_k_columns_sorted_by_k_even_with_equal_recall():
    result = {"structure": {"vector": RetrievalEvalReport(
        {5: 0.5384615384615384, 3: 0.38461538461538464, 1: 0.38461538461538464},
        0.4153846153846154,
    )}}
    lines = format_comparison_table(result).splitlines()
    header = lines[0].split()
    body = lines[1].split()

    assert header == ["Strategy", "Mode", "k(1)", "k(3)", "k(5)", "MRR"]
    assert body == ["structure", "vector", "0.38", "0.38", "0.54", "0.42"]


def test_column_count_follows_number_of_k():
    result = {"structure": {"vector": RetrievalEvalReport({5: 0.54, 10: 0.77, 1: 0.31, 3: 0.46}, 0.42)}}
    lines = format_comparison_table(result).splitlines()
    header = lines[0].split()
    body = lines[1].split()

    assert header == ["Strategy", "Mode", "k(1)", "k(3)", "k(5)", "k(10)", "MRR"]
    assert body == ["structure", "vector", "0.31", "0.46", "0.54", "0.77", "0.42"]


def test_one_line_per_strategy_mode_pair_in_dict_order():
    result = {
        "structure": {
            "hybrid": RetrievalEvalReport({5: 0.54, 3: 0.38, 1: 0.38}, 0.42),
            "vector": RetrievalEvalReport({5: 0.77, 3: 0.41, 1: 0.38}, 0.40)
        },
        "fixed": {
            "hybrid": RetrievalEvalReport({5: 0.42, 3: 0.38, 1: 0.38}, 0.38),
            "vector": RetrievalEvalReport({5: 0.54, 3: 0.41, 1: 0.41}, 0.46)
        },
    }
    lines = format_comparison_table(result).splitlines()
    header = lines[0].split()
    strategy_mode_pairs = [[text[0], text[1]] for text in (line.split() for line in lines[1:])]

    assert header == ["Strategy", "Mode", "k(1)", "k(3)", "k(5)", "MRR"]
    assert strategy_mode_pairs == [["structure", "hybrid"], ["structure", "vector"], ["fixed", "hybrid"], ["fixed", "vector"]]


def test_empty_results():
    result = {}
    lines = format_comparison_table(result).splitlines()

    assert lines == ["Results do not exist."]
