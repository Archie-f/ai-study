from rag_notes.eval_report import build_full_eval_html
from rag_notes.models import RetrievalEvalReport, JudgeResult


def test_build_full_eval_html_includes_expected_content():
    mode_reports = {
        "vector": RetrievalEvalReport(recall_k={5: 0.6}, mrr=0.42),
        "hybrid": RetrievalEvalReport(recall_k={5: 0.54}, mrr=0.40),
    }
    judge_results = [
        JudgeResult(
            question="What is RAG?",
            answer="Retrieval-augmented generation.",
            score=1.0,
            reason="Fully supported by the context.",
            passed=True,
        ),
        JudgeResult(
            question="What is BM25?",
            answer="A ranking function.",
            score=0.33,
            reason="Missing key facts from the reference answer.",
            passed=False,
        ),
    ]

    html = build_full_eval_html(mode_reports, judge_results)

    assert "Vector" in html
    assert "Hybrid" in html
    assert "PASS" in html
    assert "FAIL" in html
    assert "resultFilter" in html
    assert "What is RAG?" in html
    assert "What is BM25?" in html
