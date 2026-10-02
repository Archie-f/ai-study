from dataclasses import dataclass


@dataclass
class JobPosting:
    company: str
    title: str
    description: str


NO_MATCHING_JOBS_TOKEN = "NO_MATCHING_JOBS_FOUND"
GUARDRAIL_ANSWER = "No matching jobs found."


def build_job_context(results: list[tuple[float, JobPosting]]) -> str:
    """Turn ranked search() results into one labeled context string.

    Each result becomes a block:
        [Listing N: <company> — <title>]
        <description>

    Args:
        results: Ranked (score, JobPosting) tuples.

    Returns:
        A single string with one labeled block per result, blocks
        separated by a blank line, nothing trailing after the last one.

    Raises:
        ValueError: if results is empty — there's no context to
            build, and callers need to know that explicitly rather
            than silently getting an empty string.
    """
    if not results:
        raise ValueError("No results exist.")

    context = []
    for index, (_, job_posting) in enumerate(results, start=1):
        company = job_posting.company
        title = job_posting.title
        description = job_posting.description
        context.append(f"[Listing {index:02d}: {company} — {title}]\n{description}")

    return "\n\n".join(context)


def normalize_job_answer(text: str) -> str:
    """Convert a raw model answer into the guardrail-safe answer if needed.

        Args:
            text: The raw text returned by generate_answer().

        Returns:
            GUARDRAIL_ANSWER if NO_MATCHING_JOBS_TOKEN appears anywhere in text,
            otherwise text unchanged.
        """
    return GUARDRAIL_ANSWER if NO_MATCHING_JOBS_TOKEN in text else text
