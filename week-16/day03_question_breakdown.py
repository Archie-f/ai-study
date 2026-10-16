"""Per-question retrieval breakdown at chunk level (Week 16 Day 03, measurement only).

Run from rag-notes-qa/ with the venv active:
    python ../week-16/day03_question_breakdown.py

For every golden question and every mode it finds the rank (1-5) of the first
chunk that contains the answer_key, then reports recall@1 / recall@5 / MRR for
several groups of questions: all, by notes tag, and by where the key sits in
its chunk (inside or beyond the first ~1000 characters the embedder reads).
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())  # so "config" resolves when run as a file path

from config import PERSIST_PATH, EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import ChromaVectorStore, StructureChunker
from rag_notes.retrieval import build_retrieval_index, search
from rag_notes.retrieval_metrics import find_relevant_chunk_ids, normalize_text

MODES = ["hybrid", "vector", "bm25"]
EMBEDDER_WINDOW_CHARS = 1000  # all-MiniLM-L6-v2 reads ~256 tokens, about 1000 characters


def first_hit_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> int | None:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return rank
    return None


def summarize(ranks: list[int | None]) -> str:
    n = len(ranks)
    r1 = sum(1 for r in ranks if r == 1) / n
    r5 = sum(1 for r in ranks if r is not None) / n
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return f"n={n:2}  r@1 {r1:.2f}  r@5 {r5:.2f}  MRR {mrr:.3f}"


def main() -> None:
    with open(GOLDEN_QA_PATH) as file:
        entries = json.load(file)

    index = build_retrieval_index(EVAL_CORPUS_PATH, ChromaVectorStore(PERSIST_PATH), StructureChunker())
    chunks = index.bm25_index.chunks
    chunk_by_id = {f"{c.source.title}-{c.chunk_index}": c for c in chunks}

    groups: dict[str, list[str]] = {"all": [], "paraphrase": [], "lexical": [], "original 13": [],
                                    "key within window": [], "key beyond window": []}
    relevant: dict[str, set[str]] = {}
    for entry in entries:
        qid = entry["id"]
        relevant[qid] = find_relevant_chunk_ids(chunks, entry["answer_key"])
        groups["all"].append(qid)
        tag = entry.get("notes", "")
        groups[tag if tag in ("paraphrase", "lexical") else "original 13"].append(qid)
        chunk_text = normalize_text(chunk_by_id[sorted(relevant[qid])[0]].text)
        offset = chunk_text.find(normalize_text(entry["answer_key"]))
        groups["key beyond window" if offset > EMBEDDER_WINDOW_CHARS else "key within window"].append(qid)

    ranks: dict[str, dict[str, int | None]] = {mode: {} for mode in MODES}
    for mode in MODES:
        for entry in entries:
            results = search(index, entry["question"], 5, mode=mode)
            ranks[mode][entry["id"]] = first_hit_rank([r[0] for r in results], relevant[entry["id"]])

    for group, qids in groups.items():
        print(f"\n== {group}")
        for mode in MODES:
            print(f"   {mode:7} {summarize([ranks[mode][q] for q in qids])}")

    print("\nPer question (rank of first relevant chunk, '-' = not in top 5):")
    print("   id     hybrid vector bm25  tag")
    for entry in entries:
        q = entry["id"]
        cells = [str(ranks[m][q]) if ranks[m][q] else "-" for m in MODES]
        tag = entry.get("notes", "") if entry.get("notes", "") in ("paraphrase", "lexical") else "original"
        print(f"   {q}   {cells[0]:>5} {cells[1]:>6} {cells[2]:>4}  {tag}")


if __name__ == "__main__":
    main()
