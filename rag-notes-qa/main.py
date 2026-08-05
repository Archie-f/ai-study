import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_notes.loader import load_corpus
from rag_notes.structure_chunker import chunk_document

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--notes-root", default=os.getenv("NOTES_ROOT"))
args = parser.parse_args()
print(args.notes_root)

boundary_styles = {"Heading 1", "Heading 2", "Heading 3"}
all_chunks = []

documents = load_corpus(Path(args.notes_root))
for document in documents:
    chunks = chunk_document(document, boundary_styles)
    all_chunks.append(chunks)

print(len(documents))
print(len(all_chunks))
assert len(all_chunks) == len(documents)
print(sum(len(chunks) for chunks in all_chunks))
print(all_chunks[0][:2])
