from dataclasses import replace

from rag_notes.embedder import count_wordpiece_tokens
from rag_notes.fixed_chunker import SAFE_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_fixed_size, chunk_fixed_size_document
from rag_notes.models import SourceDocument, Chunk
from rag_notes.structure_chunker import BOUNDARY_STYLES, chunk_document


def chunk_document_structure_then_size(
        document: SourceDocument,
        tokenizer,
        limit: int,
        boundary_styles: set[str] = BOUNDARY_STYLES,
        n: int = SAFE_CHUNK_SIZE,
        o: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    """Chunk one Document into structure-aware chunks first.
    Then re-chunk the chunks which have number of tokens more than the given limit.

    Args:
        document (SourceDocument): The document to chunk.
        tokenizer: The tokenizer to calculate the wordpiece count.
        limit (int): Usable embedder window in WordPiece tokens, with [CLS] and [SEP] already subtracted (e.g. 254).
                     Chunks with more WordPiece tokens than this are re-chunked.
        boundary_styles (set[str]): The set of styles to split the document.
        n (int): Size of each re-chunked piece, counted in tiktoken tokens (not WordPiece).
        o (int): Overlap between neighboring pieces, counted in tiktoken tokens.
    Returns:
        list of Chunks with re-indexed continuously starting from 0
    """
    structure_chunks = chunk_document(document, boundary_styles)
    final_chunks_list = []

    for chunk in structure_chunks:
        wordpiece_count = count_wordpiece_tokens(chunk.text, tokenizer)
        if wordpiece_count <= limit:
            final_chunks_list.append(chunk)
        else:
            fixed_size_split_texts = chunk_fixed_size(chunk.text, n=n, o=o)
            sub_chunks = chunk_fixed_size_document(
                fixed_size_split_texts,
                chunk.source,
                chunk.heading
            )
            final_chunks_list.extend(sub_chunks)

    return [replace(chunk, chunk_index=index) for index, chunk in enumerate(final_chunks_list)]
