from pathlib import Path

import tiktoken

from rag_notes.models import SourceDocument, Chunk, DocumentMetadata

BOUNDARY_STYLES = {"Heading 1", "Heading 2", "Heading 3"}
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

encoding = tiktoken.get_encoding("cl100k_base")


def chunk_document(document: SourceDocument, boundary_styles: set[str]) -> list[Chunk]:
    """Split one Document into structure-aware chunks along its own heading boundaries.

    Args:
        document: Document to split
        boundary_styles: set of styles to split into chunks
    Returns:
        list of Chunks
    """
    chunks: list[Chunk] = []
    current_heading: str | None = None
    current_text: str = ""
    current_chunk_index: int = 0

    for paragraph in document.paragraphs:
        if paragraph[1] in boundary_styles:
            if current_text != "":
                chunks.append(Chunk(
                    text=current_text,
                    source=document.metadata,
                    heading=current_heading,
                    chunk_index=current_chunk_index,
                ))
                current_chunk_index += 1
                current_text = ""
            current_heading = paragraph[0]
        else:
            current_text += paragraph[0] + "\n"

    if current_text != "":
        chunks.append(Chunk(
            text=current_text,
            source=document.metadata,
            heading=current_heading,
            chunk_index=current_chunk_index,
        ))
    return chunks


def chunk_fixed_size(text: str, n: int, o: int) -> list[str]:
    """Split text into fixed-size, overlapping chunk texts by token count."""
    if o >= n:
        raise ValueError(f"Overlap value '{o}' can not be greater than chunk size value '{n}'")

    tokens = encoding.encode(text)
    chunk_texts = []

    for i in range(0, len(tokens), n-o):
        chunk_tokens = tokens[i:(i+n)]
        if i > 0 and len(chunk_tokens) <= o:
            continue

        chunk_texts.append(encoding.decode(chunk_tokens))

    return chunk_texts


def chunk_fixed_size_document(
        chunk_texts: list[str],
        metadata: DocumentMetadata,
        heading: str | None = None
) -> list[Chunk]:
    """Creates Chunk using each element of list."""
    return [Chunk(
        text=chunk_text,
        source=metadata,
        heading=heading,
        chunk_index=index,
    ) for index, chunk_text in enumerate(chunk_texts)]


if __name__ == "__main__":
    starts_with_body_text = [("body0", "Normal"), ("H1", "Heading 1"), ("body1", "Normal"), ("H2", "Heading 2")]
    headings_only_no_body = [("H1", "Heading 1"), ("H2", "Heading 2"), ("H3", "Heading 3")]
    body_only_no_headings = [("p1", "Normal"), ("p2", "Normal"), ("p3", "Normal")]
    single_paragraph_just_a_heading = [("H1", "Heading 1")]
    single_paragraph_just_body_text = [("p1", "Normal")]
    empty_document = []
    style_out_of_scope = [("H1", "Heading 1"), ("Sub", "Heading 4"), ("body1", "Normal")]
    two_consecutive_identical = [("H1", "Heading 1"), ("", "Normal"), ("", "Normal"), ("body1", "Normal")]
    same_heading_used_twice = [("Intro", "Heading 1"), ("body1", "Normal"), ("Intro", "Heading 1"), ("body2", "Normal")]

    docs = [
        starts_with_body_text,
        headings_only_no_body,
        body_only_no_headings,
        single_paragraph_just_a_heading,
        single_paragraph_just_body_text,
        empty_document,
        style_out_of_scope,
        two_consecutive_identical,
        same_heading_used_twice
    ]

    tickets = [
        "Can't log into my account, password reset email never arrives.",
        "App crashes every time I try to upload a photo.",
        "How do I cancel my subscription before the renewal date?",
    ]

    metadata = DocumentMetadata(week=1, day=1, file_path=Path("test.docx"), title="test")
    for doc in docs:
        source_document = SourceDocument(metadata=metadata, paragraphs=doc)
        result = chunk_document(source_document, BOUNDARY_STYLES)
        for chunk in result:
            print(chunk)

    print("-" * 40)
    chunks = chunk_fixed_size_document(tickets, metadata)
    for chunk in chunks:
        print(chunk)






