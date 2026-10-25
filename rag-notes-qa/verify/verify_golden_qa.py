import json

from config import EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import StructureThenSizeChunker
from rag_notes.embedder import load_embedding_model
from rag_notes.golden_qa import validate_golden_entries
from rag_notes.retrieval import collect_chunks


def main():
    with open(GOLDEN_QA_PATH, "r") as golden_qa_file:
        entries = json.load(golden_qa_file)

    model = load_embedding_model()
    limit = model.max_seq_length - 2
    chunker = StructureThenSizeChunker(
        tokenizer=model.tokenizer,
        limit=limit
    )
    chunks = collect_chunks(EVAL_CORPUS_PATH, chunker)
    problems = validate_golden_entries(entries, chunks)
    print(f"{len(entries)} entries, {len(chunks)} chunks, {len(problems)} problem(s)")
    for problem in problems:
        print(f"  {problem}")
    print(f"limit: {limit}")


if __name__ == "__main__":
    main()
