import re
from pathlib import Path
from docx import Document as DocxFile

from rag_notes.models import DocumentMetadata, SourceDocument


_PATTERN = re.compile(r"week-(\d+)-day-(\d+)-takeaway-notes\.docx")
KNOWN_UNSUPPORTED = {".doc", ".pdf"}


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

PARSERS = {
    ".docx": iter_paragraphs,
}

def load_corpus(notes_root: Path) -> list[SourceDocument]:
    """Walk notes_root and load every in-scope, parseable file into a SourceDocument.

    In scope: week-XX/week-XX-day-YY-takeaway-notes.docx and
    AI-Study-Comprehensive-Notes.docx; _reference/ is excluded.

    Parseable formats are driven by PARSERS (suffix -> parser function),
    so adding a new format is a one-line addition, not a rewrite.
    KNOWN_UNSUPPORTED formats (.doc, .pdf) print a warning instead of
    being silently dropped; anything else is skipped without comment.

    Args:
        notes_root (Path): The root directory of the corpus.

    Returns:
        List of SourceDocument objects that are in the corpus.
    """
    comprehensive_notes: str = "AI-Study-Comprehensive-Notes.docx"
    documents: list[SourceDocument] = []

    for filename in notes_root.rglob("*"):
        file_name = filename.name
        file_suffix = filename.suffix
        is_comprehensive_notes = file_name == comprehensive_notes

        if "_reference" in filename.parts:
            continue

        if file_suffix in PARSERS:
            parser = PARSERS.get(file_suffix)
        elif file_suffix in KNOWN_UNSUPPORTED:
            print(f"Skipping unsupported file {filename}")
            continue
        else:
            continue

        if is_comprehensive_notes or _PATTERN.match(file_name):
            week_day_data = parse_week_day(file_name)
            metadata = DocumentMetadata(
                week=week_day_data[0]  if not is_comprehensive_notes else None,
                day=week_day_data[1]  if not is_comprehensive_notes else None,
                file_path=filename,
                title=filename.stem,
            )
            document = SourceDocument(
                metadata=metadata,
                paragraphs=parser(str(filename)),
            )
            documents.append(document)

    return documents
