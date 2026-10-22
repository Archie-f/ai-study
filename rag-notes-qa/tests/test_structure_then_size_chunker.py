from pathlib import Path

from rag_notes.adapters import StructureThenSizeChunker
from rag_notes.fixed_chunker import chunk_text_fixed_size
from rag_notes.models import SourceDocument, DocumentMetadata
from rag_notes.structure_chunker import chunk_document


class FakeTokenizer:
    def tokenize(self, text):
        return text.split()      # one word = one token


boundary_styles = {"Heading 2", "Heading 3"}
document = SourceDocument(
        metadata=DocumentMetadata(week=9, day=2, file_path=Path("fake.docx"), title="fake"),
        paragraphs=[
            ("A", "Heading 2"),
            ("The first paragraph", "Normal"),
            ("B", "Heading 3"),
            ("This is the second paragraph that is supposed to have twelve words.", "Normal"),
            ("C", "Heading 3"),
            ("Last paragraph.", "Normal")
        ]
    )


def test_short_section_stays_whole():
    """A section whose WordPiece count is within the limit stays one chunk with its text untouched."""
    tokenizer = FakeTokenizer()
    chunker = StructureThenSizeChunker(
        tokenizer=tokenizer,
        limit=5,
        boundary_styles=boundary_styles,
        n=5,
        o=2
    )
    result = chunker.chunk(document)

    assert result[0].text == "The first paragraph\n"
    assert result[-1].text == "Last paragraph.\n"


def test_long_section_is_cut_and_pieces_keep_heading_and_source():
    """A section over the limit becomes several pieces; every piece keeps the heading and source of its section."""
    tokenizer = FakeTokenizer()
    chunker = StructureThenSizeChunker(
        tokenizer=tokenizer,
        limit=5,
        boundary_styles=boundary_styles,
        n=5,
        o=2
    )
    result = chunker.chunk(document)
    expected_pieces_length = len(chunk_text_fixed_size(chunk_document(document, boundary_styles)[1].text, n=5, o=2))

    assert len(result) == 1 + expected_pieces_length + 1
    for chunk in result[1:-1]:
        assert chunk.heading == "B"
        assert chunk.source == document.metadata


def test_chunk_index_is_continuous_from_zero():
    """Kept chunks and pieces together are numbered 0, 1, 2, ... with no gap and no repeat."""
    tokenizer = FakeTokenizer()
    chunker = StructureThenSizeChunker(
        tokenizer=tokenizer,
        limit=5,
        boundary_styles=boundary_styles,
        n=5,
        o=2
    )
    result_indexes = [chunk.chunk_index for chunk in chunker.chunk(document)]

    assert result_indexes == list(range(len(result_indexes)))


def test_section_exactly_at_limit_stays_whole():
    """A section with exactly `limit` WordPiece tokens stays one chunk (the rule is <=, not <)."""
    document_1 = SourceDocument(
        metadata=DocumentMetadata(week=9, day=2, file_path=Path("fake.docx"), title="fake"),
        paragraphs=[
            ("A", "Heading 2"),
            ("A paragraph with five words.", "Normal")
        ]
    )
    tokenizer = FakeTokenizer()
    chunker = StructureThenSizeChunker(
        tokenizer=tokenizer,
        limit=5,
        boundary_styles=boundary_styles,
        n=3,
        o=2
    )
    result = chunker.chunk(document_1)
    assert len(result) == 1
    assert result[0].text == "A paragraph with five words.\n"
