from rag_notes.structure_chunker import chunk_document
from dataclasses import dataclass

@dataclass
class SourceDocument:
    metadata: object
    paragraphs: list

test1 = SourceDocument(
    metadata="meta",
    paragraphs=[
        ("This is the first paragraph.", "Normal"),
        ("This is the second paragraph.", "None"),
        ("1. Chunks", "Heading 2"),
        ("1.1 What Is a Chunk", "Heading 3"),
        ("A chunk is a piece of text.", "Normal"),
        ("Splitting by fixed size ignores structure.", "Normal"),
        ("1.2 Corpus Scope", "Heading 3"),
        ("This was locked yesterday.", "Normal"),
        ("1.3 Examples", "Heading 3"),
    ]
)

result = chunk_document(test1, {"Heading 2", "Heading 3"})
for c in result:
    print(c)