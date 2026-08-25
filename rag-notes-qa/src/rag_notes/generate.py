from rag_notes.citations import build_source_label
from rag_notes.models import Chunk
from llm_compare.providers.base import LLMProvider, LLMResult


SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain enough information to answer, "
    "reply only as 'I don't know' instead of guessing or using outside knowledge. "
    "When you use a source, refer to it by its [Source N: <label>] label."
)


def build_context(results: list[tuple[str, float, Chunk]]) -> str:
    """Turn ranked search() results into one labeled context string.

    Each result becomes a block:
        [Source N: <label>]
        <chunk text>

    where <label> is a citation-friendly identifier built from the
    chunk's source metadata (e.g. "week-01-day-01" or the doc title
    if week/day aren't set).

    Args:
        results: Ranked (chunk_id, score, Chunk) tuples, as returned
            by retrieval.search().

    Returns:
        A single string with one labeled block per result, blocks
        separated by a blank line, nothing trailing after the last one.

    Raises:
        ValueError: if results is empty — there's no context to
            build, and callers need to know that explicitly rather
            than silently getting an empty string.
    """
    if not results:
        raise ValueError("No results returned")

    context = []
    for index, (_, _, chunk) in enumerate(results, start=1):
        metadata = chunk.source
        label = build_source_label(metadata)
        context.append(f"[Source {index:02d}: {label}]\n{chunk.text}")

    return "\n\n".join(context)


def generate_answer(question: str, context: str, provider: LLMProvider) -> LLMResult:
    """Answer a question using only the given context, via a reused Provider.

    Assembles a (system_prompt, user_prompt) pair — SYSTEM_PROMPT fixed,
    context + question in the user turn — and calls provider.ask() with it.

    Args:
        question: The question to answer.
        context: Pre-built context string, as returned by build_context().
        provider: An LLMProvider instance (e.g. OllamaProvider(),
            AnthropicProvider()) to call.

    Returns:
        The provider's LLMResult (text, cost, tokens, latency).
    """
    user_prompt = f"{context.strip()}\n\nQuestion: {question.strip()}"
    return provider.ask(user_input=user_prompt, system_prompt=SYSTEM_PROMPT)
