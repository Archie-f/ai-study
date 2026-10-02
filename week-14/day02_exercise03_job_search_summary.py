from dataclasses import dataclass


@dataclass
class MatchReport:
    mrr: float
    precision_at_k: dict[int, float]


@dataclass
class ApplicationResult:
    got_interview: bool


def build_job_search_summary(strategy_reports: dict[str, MatchReport], application_results: list[ApplicationResult]) -> dict:
    result_dict = {}
    mrr_per_strategy = {}
    precision_per_strategy = {}
    for strategy, report in strategy_reports.items():
        mrr_per_strategy[strategy] = report.mrr
        precision_per_strategy[strategy] = report.precision_at_k
    result_dict["mrr_per_strategy"] = mrr_per_strategy
    result_dict["precision_per_strategy"] = precision_per_strategy

    got = sum(1 for result in application_results if result.got_interview)
    total = len(application_results)
    result_dict["interview_rate"] = 0 if total == 0 else got / total

    return result_dict
