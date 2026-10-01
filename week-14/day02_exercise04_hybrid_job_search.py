from dataclasses import dataclass
from typing import Literal

from rag_notes.hybrid_search import build_rank_map, rrf_merge


@dataclass
class JobPosting:
    job_id: str
    title: str
    keywords: list[str]
    posted_days_ago: int


def keyword_match_search(query: str, jobs: list[JobPosting], n: int) -> list[tuple[str, float, JobPosting]]:
    """Score jobs by how many query words overlap with the job's keywords.
    (Simplified stand-in for bm25_search() — matches on words.)

    Example: query="python backend", a job with keywords=["python","django","backend"]
    matches 2 words -> score 2.0.
    """
    query_terms = set(query.lower().split())
    scored = []
    for job in jobs:
        job_terms = set(k.lower() for k in job.keywords)
        score = float(len(query_terms & job_terms))
        if score > 0:
            scored.append((job.job_id, score, job))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:n]


def recency_search(jobs: list[JobPosting], n: int) -> list[tuple[str, float, JobPosting]]:
    """Rank jobs by how recently they were posted; newer gets a higher score.
    (Stand-in for get_query_result() — ranks by freshness, not keywords.)

    Example: posted_days_ago=1 scores higher than posted_days_ago=10.
    """
    scored = [(job.job_id, 1.0 / (1 + job.posted_days_ago), job) for job in jobs]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:n]


def hybrid_job_search(
        query: str,
        jobs: list[JobPosting],
        n: int = 3,
        mode: Literal["hybrid", "keyword", "recency"] = "hybrid"
) -> list[tuple[str, float, JobPosting]]:
    """Search job postings with a selectable ranking mode.

    Args:
        query: search text, used only by keyword_match_search
        jobs: postings to search over
        n: how many results to return
        mode: "keyword" -> keyword_match_search only, "recency" -> recency_search
            only, "hybrid" (default) -> RRF-fuse both rankers

    Returns:
        list of (job_id, score, job) tuples, highest score first, length <= n
    """
    if mode == "keyword":
        keyword_results = keyword_match_search(query, jobs, n)
        return keyword_results

    if mode == "recency":
        recency_results = recency_search(jobs, n)
        return recency_results


    keyword_ids = [job_id for job_id, _, _ in keyword_match_search(query, jobs, n)]
    recency_ids = [job_id for job_id, _, _ in recency_search(jobs, n)]
    id_job_dict = {job.job_id: job for job in jobs}

    ranked_maps = [build_rank_map(keyword_ids), build_rank_map(recency_ids)]
    merged = rrf_merge(ranked_maps)

    return [
        (job_id, score, id_job_dict[job_id])
        for job_id, score in merged[:n]
    ]

