from pathlib import Path

from config import EVAL_CORPUS_PATH
from rag_notes.adapters import StructureChunker
from rag_notes.embedder import wordpiece_counts, load_embedding_model
from rag_notes.loader import load_corpus
from rag_notes.protocols import Chunker


def collect_chunk_counts(
        corpus_path: Path,
        chunker: Chunker,
        tokenizer,
) -> list[int]:
    """Chunk every document in the corpus and count each chunk's WordPiece tokens.

    Args:
        corpus_path: root folder of the corpus (e.g. EVAL_CORPUS_PATH)
        chunker: any Chunker (e.g. StructureChunker())
        tokenizer: the embedder's tokenizer (model.tokenizer)
    Returns:
        one count per chunk, all documents together, in document order
    """
    source_documents = load_corpus(corpus_path)
    chunks = []
    for document in source_documents:
        chunks.extend(chunker.chunk(document))
    return wordpiece_counts(chunks, tokenizer)


def bucket_counts(
        chunk_counts: list[int],
        edges: list[int],
) -> list[int]:
    """Count how many chunk token counts fall into each bucket.
    Each count goes into the first bucket whose edge it does not exceed (edges are inclusive).

    Args:
        chunk_counts: WordPiece token count of each chunk
        edges: sorted upper limits of the buckets, e.g. [100, 200]
    Returns:
        len(edges) + 1 counts; the last one is for values above the last edge,
        e.g. bucket_counts([50, 300], [100, 200]) -> [1, 0, 1]
    """
    counts = [0] * (len(edges) + 1)
    for chunk_count in chunk_counts:
        bucket = len(edges)
        for index, edge in enumerate(edges):
            if chunk_count <= edge:
                bucket = index
                break
        counts[bucket] += 1
    return counts


def main():
    model = load_embedding_model()
    tokenizer = model.tokenizer

    chunk_counts = collect_chunk_counts(
        corpus_path=EVAL_CORPUS_PATH,
        chunker=StructureChunker(),
        tokenizer=tokenizer
    )

    limit = model.max_seq_length - 2
    edges = [128, limit, 512]

    counts = bucket_counts(chunk_counts, edges)
    print(f"chunks: {len(chunk_counts)} over {limit}: {sum(1 for c in chunk_counts if c > limit)}"
          f"\nlargest: {max(chunk_counts)}"
          f"\n<=128: {counts[0]}\t\t129-{limit}: {counts[1]}\t\t{limit + 1}-512: {counts[2]}\t\t>512: {counts[3]}")


if __name__ == "__main__":
    main()
