from sentence_transformers import CrossEncoder

from config import PERSIST_PATH, EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import StructureThenSizeChunker, ChromaVectorStore, CrossEncoderReranker
from rag_notes.embedder import load_embedding_model
from rag_notes.models import RetrievalIndex
from rag_notes.protocols import Reranker
from rag_notes.retrieval import build_retrieval_index
from verify.verify_chunking_eval import format_comparison_table
from verify.verify_retrieval_eval import compare_retrieval_modes


def get_retrieval_index_and_reranker() -> tuple[RetrievalIndex, Reranker]:
    model = load_embedding_model()
    limit = model.max_seq_length - 2
    chunker = StructureThenSizeChunker(
        tokenizer=model.tokenizer,
        limit=limit,
    )
    vector_store = ChromaVectorStore(PERSIST_PATH)
    retrieval_index = build_retrieval_index(
        notes_root=EVAL_CORPUS_PATH,
        vector_store=vector_store,
        chunker=chunker,
    )
    reranker = CrossEncoderReranker(CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"))

    return retrieval_index, reranker


def main():
    retrieval_index, reranker = get_retrieval_index_and_reranker()

    baseline = compare_retrieval_modes(retrieval_index, GOLDEN_QA_PATH, level="chunk")
    reranked = compare_retrieval_modes(retrieval_index, GOLDEN_QA_PATH, level="chunk", reranker=reranker)

    print(format_comparison_table({"struct+sz": baseline, "+rerank": reranked}))


if __name__ == "__main__":
    main()
