import tiktoken

from rag_notes.models import Chunk
from rag_notes.models import DocumentMetadata

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

encoding = tiktoken.get_encoding("cl100k_base")


def chunk_fixed_size(text: str, n: int, o: int) -> list[str]:
    """Split text into fixed-size, overlapping chunk texts by token count.
        Arguments:
            text: text to split
            n: number of tokens to split
            o: overlapping token count
        Returns:
            list of chunk texts
    """
    if o >= n:
        raise ValueError(f"Overlap value '{o}' can not be greater than chunk size value '{n}'")

    chunk_texts = []
    tokens = encoding.encode(text)

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
    """Creates Chunk using each element of list.

        Arguments:
            chunk_texts: list of chunk texts
            metadata: document metadata
            heading: heading of the chunk
        Returns:
            list of Chunk
    """
    return [Chunk(
        text=chunk_text,
        source=metadata,
        heading=heading,
        chunk_index=index,
    ) for index, chunk_text in enumerate(chunk_texts)]
