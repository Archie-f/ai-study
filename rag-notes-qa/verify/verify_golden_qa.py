import json

from config import EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import StructureChunker
from rag_notes.golden_qa import validate_golden_entries
from rag_notes.loader import load_corpus


def main():
    with open(GOLDEN_QA_PATH, "r") as golden_qa_file:
        entries = json.load(golden_qa_file)

    chunker = StructureChunker()
    chunks = [chunk for document in load_corpus(EVAL_CORPUS_PATH) for chunk in chunker.chunk(document)]

    problems = validate_golden_entries(entries, chunks)
    print(f"{len(entries)} entries, {len(chunks)} chunks, {len(problems)} problem(s)")
    for problem in problems:
        print(f"  {problem}")


if __name__ == "__main__":
    main()
