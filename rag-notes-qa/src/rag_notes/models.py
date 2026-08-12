from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentMetadata:
    """Source information for one loaded document."""
    week: int | None
    day: int | None
    file_path: Path
    title: str


@dataclass
class SourceDocument:
    """A single loaded source document with metadata and raw paragraphs plus style information."""
    metadata: DocumentMetadata
    paragraphs: list[tuple[str, str]]


@dataclass
class Chunk:
    """A single chunk with enough metadata to cite its source, produced by either chunker."""
    text: str
    source: DocumentMetadata
    heading: str | None
    chunk_index: int


@dataclass
class EmbeddedChunk:
    """A Chunk paired with the embedding vector produced from its text."""
    chunk: Chunk
    vector: list[float]


@dataclass
class BM25Index:
    """Everything needed to score a query against a corpus of Chunks."""
    chunks: list[Chunk]
    tokenized_docs: list[list[str]]
    doc_freq: Counter
    avgdl: float
    k1: float = 1.2
    b: float = 0.75
