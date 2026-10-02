from llm_compare.providers.base import LLMProvider, LLMResult
from rag_notes.citations import build_source_label, build_citations, format_citations
from rag_notes.generate import SYSTEM_PROMPT
from rag_notes.models import Chunk, RetrievalIndex, AnsweredQuery
from rag_notes.retrieval import search

NO_ANSWER_TOKEN = "NO_ANSWER_FOUND"
GUARDRAIL_ANSWER = "I don't know"


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
        raise ValueError("No results exist.")

    context = []
    for index, (_,_,chunk) in enumerate(results, start=1):
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
    user_input = f"Context: {context.strip()}\nQuestion: {question.strip()}\n"
    return provider.ask(user_input, SYSTEM_PROMPT)


def normalize_answer(text: str) -> str:
    """Convert a raw model answer into the guardrail-safe answer if needed.

    Args:
        text: The raw text returned by generate_answer().

    Returns:
        GUARDRAIL_ANSWER if NO_ANSWER_TOKEN appears anywhere in text,
        otherwise text unchanged.
    """
    return GUARDRAIL_ANSWER if NO_ANSWER_TOKEN in text else text


def answer_question(index: RetrievalIndex, question: str, provider: LLMProvider, n: int = 5) -> AnsweredQuery:
    """Run the full pipeline — retrieval, generation, citations — for one question.

    If build_context() raises ValueError (no results to build context from),
    skip generation entirely and return the guardrail answer with no citations
    instead of letting the exception propagate.

    Args:
        index: A RetrievalIndex from build_retrieval_index().
        question: The question to answer.
        provider: An LLMProvider instance to generate the answer with.
        n: How many search() results to retrieve and cite.

    Returns:
        An AnsweredQuery bundling the question, the generated answer,
        and the citations backing it — citations empty when the
        guardrail fired before generation.
    """
    results = search(index, question, n=n)

    try:
        context = build_context(results)
    except ValueError:
        return AnsweredQuery(
            query=question,
            answer=GUARDRAIL_ANSWER,
            citations=[]
        )

    llm_result = generate_answer(question, context, provider)
    normalized_answer = normalize_answer(llm_result.text)
    citations = build_citations(results)

    return AnsweredQuery(
        query=question,
        answer=normalized_answer,
        citations=citations
    )


def display_answer(answered: AnsweredQuery) -> str:
    """Format an AnsweredQuery as one printable, human-readable block.

    Combines the question, the answer, and format_citations()'s output
    into a single display-ready string — keeps AnsweredQuery itself
    free of any formatting concerns, same separation as format_citations().

    Args:
        answered: The AnsweredQuery to display.

    Returns:
        A multi-line block: the question, the answer, then the
        formatted citations — labeled so a reader can tell them apart
        even when the answer is the GUARDRAIL_ANSWER guardrail phrase.
    """
    if answered.answer == GUARDRAIL_ANSWER:
        sources = "No related answer found! Retrieved context:\n"
    else:
        sources = f"Sources: \n{format_citations(answered.citations)}"

    return f"Question: {answered.query}\nAnswer: {answered.answer}\n{sources}"
