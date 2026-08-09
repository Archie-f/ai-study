import os
from pathlib import Path
from dotenv import load_dotenv

from rag_notes.loader import DocumentMetadata, SourceDocument, iter_paragraphs, parse_week_day
from rag_notes.structure_chunker import BOUNDARY_STYLES, chunk_document
from rag_notes.fixed_chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, encoding, chunk_fixed_size, chunk_fixed_size_document

load_dotenv()
notes_root = os.getenv("NOTES_ROOT")
path = Path(notes_root) / "week-09" / "week-09-day-01-takeaway-notes.docx"
week_day = parse_week_day(str(path.name))

# Chunking the document by using structure aware method
document_metadata = DocumentMetadata(
    week=week_day[0],
    day=week_day[1],
    file_path=path,
    title=path.stem,
)

source_document = SourceDocument(
    metadata=document_metadata,
    paragraphs=iter_paragraphs(str(path)),
)

structure_aware_chunks = chunk_document(source_document, BOUNDARY_STYLES)

# Chunking the document by using structure aware method
document_text = "\n".join(paragraph[0] for paragraph in source_document.paragraphs)

chunked_texts = chunk_fixed_size(
    document_text,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP
)

fixed_size_chunks = chunk_fixed_size_document(
    chunked_texts,
    source_document.metadata
)

# Compare the results

print("Structure Aware Chunking Chunks")
print(f"{'Chunk Index'} | {'Heading':<35} | {'Token Count':<3} | {'Text'}")
print("-" * 102)
for chunk in structure_aware_chunks:
    heading = f"{chunk.heading[:30]}..." if chunk.heading else chunk.heading
    print(f"[{chunk.chunk_index:>2}]{'':7} | {heading!r:<35} | {len(encoding.encode(chunk.text)):>4} tokens | {chunk.text[:30]!r}...")

print("=" * 102)
print("Fixed Size Chunking Chunks")
print(f"{'Chunk Index'} | {'Heading':<35} | {'Token Count':<3} | {'Text'}")
print("-" * 102)

for chunk in fixed_size_chunks:
    print(f"[{chunk.chunk_index:>2}]{'':7} | {chunk.heading!r:<35} | {len(encoding.encode(chunk.text)):>4} tokens | {chunk.text[:30]!r}...")