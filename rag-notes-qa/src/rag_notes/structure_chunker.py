from dataclasses import dataclass

from rag_notes.loader import DocumentMetadata, SourceDocument

@dataclass
class Chunk:
    """A single structure-aware chunk with enough metadata to cite its source."""
    text: str
    source: DocumentMetadata
    heading: str | None
    chunk_index: int

def chunk_document(document: SourceDocument, boundary_styles: set[str]) -> list[Chunk]:
    """Split one Document into structure-aware chunks along its own heading boundaries.
        Args:
            document: Document to split
            boundary_styles: set of styles to split into chunks
        Returns:
            list of Chunks
    """
    chunks: list[Chunk] = []
    current_heading: str | None = None
    current_text: str = ""
    current_chunk_index: int = 0

    for paragraph in document.paragraphs:
        if paragraph[1] in boundary_styles:
            if current_heading is None and not current_text:
                current_heading = paragraph[0]

            if paragraph[0] != current_heading:
                if current_text != "":
                    chunks.append(Chunk(
                        text=current_text,
                        source=document.metadata,
                        heading=current_heading,
                        chunk_index=current_chunk_index
                    ))
                    current_text = ""
                    current_chunk_index += 1

            current_heading = paragraph[0]
        else:
            current_text += paragraph[0] + "\n"

    if current_text != "":
        chunks.append(Chunk(
            text=current_text,
            source=document.metadata,
            heading=current_heading,
            chunk_index=current_chunk_index
        ))
    return chunks
