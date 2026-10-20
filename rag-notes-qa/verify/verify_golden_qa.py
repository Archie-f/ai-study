import json

from config import EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import StructureChunker
from rag_notes.golden_qa import validate_golden_entries
from rag_notes.retrieval import collect_chunks


def main():
    with open(GOLDEN_QA_PATH, "r") as golden_qa_file:
        entries = json.load(golden_qa_file)

    chunks = collect_chunks(EVAL_CORPUS_PATH, StructureChunker())
    problems = validate_golden_entries(entries, chunks)
    print(f"{len(entries)} entries, {len(chunks)} chunks, {len(problems)} problem(s)")
    for problem in problems:
        print(f"  {problem}")


if __name__ == "__main__":
    main()
