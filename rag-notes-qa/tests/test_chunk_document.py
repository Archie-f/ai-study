from pathlib import Path

from rag_notes.loader import DocumentMetadata, SourceDocument
from rag_notes.structure_chunker import chunk_document

boundary_styles = {"Heading 2", "Heading 3"}

def test_normal_two_sections():
    document = SourceDocument(
        metadata=DocumentMetadata(week=9, day=2, file_path=Path("fake.docx"), title="fake"),
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

    result = chunk_document(document, boundary_styles)

    assert len(result) == 3

    assert result[0].heading is None
    assert result[0].text == "This is the first paragraph.\nThis is the second paragraph.\n"

    assert result[1].heading == "1.1 What Is a Chunk"
    assert result[1].text == "A chunk is a piece of text.\nSplitting by fixed size ignores structure.\n"

    assert result[2].heading == "1.2 Corpus Scope"
    assert result[2].text == "This was locked yesterday.\n"

def test_back_to_back_headings():
    document = SourceDocument(
        metadata=DocumentMetadata(week=9, day=2, file_path=Path("fake2.docx"), title="fake2"),
        paragraphs=[
            ("Part 1", "Heading 2"),
            ("1.1 Something", "Heading 3"),
            ("Body text here.", "Normal"),
        ]
    )

    result = chunk_document(document, boundary_styles)
    all_headings = []

    for chunk in result:
        all_headings.append(chunk.heading)

    assert "Part 1" not in all_headings
    assert len(result) == 1
    assert result[0].heading == "1.1 Something"
    assert result[0].text == "Body text here.\n"

def test_text_before_first_heading():
    document = SourceDocument(
        metadata=DocumentMetadata(week=9, day=2, file_path=Path("fake3.docx"), title="fake3"),
        paragraphs=[
            ("Intro text before any heading.", "Normal"),
            ("1.1 What Is a Chunk", "Heading 3"),
            ("A chunk is a piece of text.", "Normal"),
        ]
    )
    result = chunk_document(document, boundary_styles)

    assert result[0].heading is None
    assert result[0].text == "Intro text before any heading.\n"
    assert len(result) == 2
    assert result[1].heading == "1.1 What Is a Chunk"
    assert result[1].text == "A chunk is a piece of text.\n"