from rag_notes.models import DocumentMetadata, Chunk, Citation


def build_source_label(metadata: DocumentMetadata) -> str:
    """Build a citation-friendly label from a chunk's source metadata.

    Args:
        metadata: The chunk's source metadata, as attached to a Chunk
            by the chunker (chunk.source).

    Returns:
        "week-WW-day-DD" (zero-padded) if both week and day are set
        on metadata, otherwise metadata.title.
    """
    if metadata.week is None or metadata.day is None:
        return metadata.title
    return f"week-{metadata.week:02d}-day-{metadata.day:02d}"


def build_citations(results: list[tuple[str, float, Chunk]]) -> list[Citation]:
    """Extract structured Citation objects from search() results.

    Mirrors build_context()'s enumeration exactly — the same results
    list, in the same order, must produce the same [Source N: label]
    numbering in both places, so a citation's source_index always
    matches the number the model actually saw in its prompt.

    Args:
        results: Ranked (chunk_id, score, Chunk) tuples, as returned
            by retrieval.search() — the same input build_context()
            consumes.

    Returns:
        One Citation per result, in the same order as results.
    """
    return [
        Citation(
            source_index=index,
            label=build_source_label(chunk.source),
            heading=chunk.heading,
            text=chunk.text,
        )
        for index, (_, _, chunk) in enumerate(results, start=1)
    ]


def format_citations(citations: list[Citation]) -> str:
    """Format citations as human-readable, display-ready text.

    Keeps display formatting separate from the Citation data itself —
    AnsweredQuery carries the structured citations, this turns them
    into something a person (or Day 03's CLI) can print.

    Args:
        citations: Citations to format, in source-number order.

    Returns:
        One line per citation, joined by newlines, nothing trailing.
        Each line should be enough to identify the source at a
        glance — source number, label, and heading (when present).
    """
    formatted_citations = []
    for citation in citations:
        heading = citation.heading if citation.heading is not None else "No heading"
        formatted_citations.append(
            f"[Source {citation.source_index:02d}: {citation.label} - {heading}]"
        )
    return "\n".join(formatted_citations)
