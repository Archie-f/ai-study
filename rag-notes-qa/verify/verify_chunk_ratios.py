from math import floor

import numpy as np

from config import EVAL_CORPUS_PATH
from rag_notes.adapters import StructureChunker
from rag_notes.embedder import load_embedding_model, chunk_ratios, wordpiece_counts
from rag_notes.fixed_chunker import encoding
from rag_notes.models import Chunk
from rag_notes.retrieval import collect_chunks


def display_chunk_ratio_table(chunks: list[Chunk], tokenizer, encoding, limit):
    """Print the mean, median, p95 and max WordPiece/tiktoken ratio of the chunks,
    each with the chunk size (in tiktoken tokens) that would fit the embedder window.

    Args:
        chunks: chunks to measure
        tokenizer: the embedder's tokenizer (model.tokenizer)
        encoding: the tiktoken encoding used by the fixed chunker
        limit: usable embedder window, e.g. 254
    """
    chunk_ratios_list = chunk_ratios(chunks, tokenizer, encoding)
    r_mean = np.mean(chunk_ratios_list)
    r_median = np.median(chunk_ratios_list)
    r_p95 = np.percentile(chunk_ratios_list, 95)
    r_max = np.max(chunk_ratios_list)

    print(f"{'Average chunk ratio:':<25} {r_mean:.2f} -> n_safe = {floor(limit / r_mean)}\n"
          f"{'Median chunk ratio:':<25} {r_median:.2f} -> n_safe = {floor(limit / r_median)}\n"
          f"{'p95 chunk ratio:':<25} {r_p95:.2f} -> n_safe = {floor(limit / r_p95)}\n"
          f"{'Max chunk ratio:':<25} {r_max:.2f} -> n_safe = {floor(limit / r_max)}\n")


def main():
    model = load_embedding_model()
    tokenizer = model.tokenizer
    chunks = collect_chunks(EVAL_CORPUS_PATH, StructureChunker())

    limit = model.max_seq_length - 2
    wordpiece_counts_list = wordpiece_counts(chunks, tokenizer)
    over_limit_chunks = [chunk for chunk, count in zip(chunks, wordpiece_counts_list) if count > limit]

    print(f"All chunks ({len(chunks)})")
    display_chunk_ratio_table(chunks, tokenizer, encoding, limit)
    print()
    print(f"Over limit ({len(over_limit_chunks)})")
    display_chunk_ratio_table(over_limit_chunks, tokenizer, encoding, limit)


if __name__ == '__main__':
    main()
