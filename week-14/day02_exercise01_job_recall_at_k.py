def job_recall_at_k(ranked_job_ids: list[str], applied_job_ids: set[str], k: int) -> float:
    """Return 1.0 if any of the top-k ranked job IDs is applied, else 0.0.

        Args:
            ranked_job_ids: job IDs in ranked order, best match first
            applied_job_ids: the set of job IDs that are applied
            k: how many of the top results to check
        Returns:
            1.0 on a hit within the top k, 0.0 otherwise
        """
    return 1 if any(ranked_job_id in applied_job_ids for ranked_job_id in ranked_job_ids[:k]) else 0.0