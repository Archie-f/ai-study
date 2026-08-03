import re
from dataclasses import dataclass
from pathlib import Path
from docx import Document as DocxFile


_PATTERN = re.compile(r"week-(\d+)-day-(\d+)-takeaway-notes\.docx")

@dataclass
class DocumentMetadata:
    """Source information for one loaded document."""
    week: int | None
    day: int | None
    file_path: Path
    title: str


@dataclass
class Document:
    """A single loaded source document with metadata and raw paragraphs plus style information."""
    metadata: DocumentMetadata
    paragraphs: list[tuple[str, str]]


def iter_paragraphs(path: str) -> list[tuple[str, str]]:
    """Yield (text, style_name) for every non-empty paragraph in a .docx file."""
    paragraphs: list[tuple[str, str]] = []
    doc = DocxFile(path)

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            style_name = paragraph.style.name if paragraph.style is not None else "Normal"
            paragraphs.append((paragraph.text.strip(), style_name))

    return paragraphs

def parse_week_day(filename: str) -> tuple[int | None, int | None]:
    """Extract (week, day) from a takeaway-notes filename.

    Args:
        filename (str): The filename of the takeaway-notes file.
    Returns:
        Number of week and day for the takeaway-notes file of the given path.
        (None, None) for the comprehensive notes file, and for any filename
        that doesn't match the takeaway-notes pattern (e.g. study-plan docs).
    """
    result = _PATTERN.match(filename)
    return (int(result.group(1)), int(result.group(2))) if result else (None, None)

def load_corpus(notes_root: Path) -> list[Document]:
    """Walk notes_root and load every in-scope .docx into a Document.

    In scope: week-XX/week-XX-day-YY-takeaway-notes.docx and
    AI-Study-Comprehensive-Notes.docx. Out of scope (skipped): everything
    under _reference/, including study-plan docs.

    Args:
        notes_root (Path): The root directory of the corpus.

    Returns:
        List of Document objects that are in the corpus.
    """
    comprehensive_notes: str = "AI-Study-Comprehensive-Notes.docx"
    documents: list[Document] = []

    for filename in notes_root.rglob("*.docx"):
        if "_reference" in filename.parts:
            continue

        file_name = filename.name
        is_comprehensive_notes = file_name == comprehensive_notes
        if is_comprehensive_notes or _PATTERN.match(file_name):
            week_day_data = parse_week_day(file_name)
            metadata = DocumentMetadata(
                week=week_day_data[0]  if not is_comprehensive_notes else None,
                day=week_day_data[1]  if not is_comprehensive_notes else None,
                file_path=filename,
                title=filename.stem,
            )
            document = Document(
                metadata=metadata,
                paragraphs=iter_paragraphs(str(filename)),
            )
            documents.append(document)

    return documents
