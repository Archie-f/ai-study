from rag_notes.models import RetrievalEvalReport, JudgeResult


def build_eval_summary(
    mode_reports: dict[str, RetrievalEvalReport],
    judge_results: list[JudgeResult],
) -> dict:
    """Compute the summary numbers build_full_eval_html() is built from.

    (Scoped-down teaching version of the real function — skips chart
    rendering and HTML/CSS assembly, keeps only the aggregation logic.)

    Args:
        mode_reports: one RetrievalEvalReport per retrieval mode
        judge_results: one JudgeResult per golden-set question
    Returns:
        {"mrr_by_mode": {...}, "recall_by_mode": {...}, "passed": int,
         "total": int, "pass_rate": float}, pass_rate is 0.0 when total is 0
    """
    result_dict = {}
    mrr_by_mode = {}
    recall_by_mode = {}
    for mode, report in mode_reports.items():
        mrr_by_mode[mode] = report.mrr
        recall_by_mode[mode] = report.recall_k
    result_dict["mrr_by_mode"] = mrr_by_mode
    result_dict["recall_by_mode"] = recall_by_mode

    passed = sum(1 for result in judge_results if result.passed == True)
    result_dict["passed"] = passed

    total = len(judge_results)
    result_dict["total"] = total

    pass_rate = 0.0 if total == 0 else passed / total
    result_dict["pass_rate"] = pass_rate

    return result_dict
