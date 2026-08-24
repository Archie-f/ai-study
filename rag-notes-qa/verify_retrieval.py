import os
from pathlib import Path

from dotenv import load_dotenv

from rag_notes.retrieval import build_retrieval_index, search


load_dotenv()

notes_root_env = os.getenv("NOTES_ROOT")
if notes_root_env is None:
    raise RuntimeError("NOTES_ROOT not set — check your .env file")
NOTES_ROOT = Path(notes_root_env)
PERSIST_PATH = str(Path(__file__).parent / "persistent")
QUERY = "Why do type hints matter in Python even though the language doesn't enforce them at runtime?"


def main() -> None:
    index = build_retrieval_index(NOTES_ROOT, PERSIST_PATH)
    results = search(index, QUERY, n=5)
    for rank, (chunk_id, score, chunk) in enumerate(results, start=1):
        heading = chunk.heading or "(no heading)"
        print(f"Rank: {rank} | Score: {score:.6f}| Heading: {heading}")
        # ... prints rank, score, heading, and flags the expected match





if __name__ == "__main__":
    main()