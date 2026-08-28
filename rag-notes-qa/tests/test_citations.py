from pathlib import Path

from rag_notes.citations import build_source_label, build_citations, format_citations
from rag_notes.models import Chunk, Citation, DocumentMetadata

_dated_metadata = DocumentMetadata(week=9, day=1, file_path=Path("fake.docx"), title="fake")
_undated_metadata = DocumentMetadata(week=None, day=None, file_path=Path("fake.docx"), title="Untitled Notes")

chunk_with_heading = Chunk(
    text="Type hints help readability.",
    source=_dated_metadata,
    heading="Type Hints",
    chunk_index=0,
)
chunk_without_heading = Chunk(
    text="Napoleon lost at Waterloo.",
    source=_undated_metadata,
    heading=None,
    chunk_index=0,
)


def test_build_source_label_with_week_and_day():
    assert build_source_label(_dated_metadata) == "week-09-day-01"


def test_build_source_label_falls_back_to_title_when_week_missing():
    metadata = DocumentMetadata(week=None, day=1, file_path=Path("fake.docx"), title="fallback title")
    assert build_source_label(metadata) == "fallback title"


def test_build_source_label_falls_back_to_title_when_day_missing():
    metadata = DocumentMetadata(week=9, day=None, file_path=Path("fake.docx"), title="fallback title")
    assert build_source_label(metadata) == "fallback title"


def test_build_source_label_falls_back_to_title_when_both_missing():
    assert build_source_label(_undated_metadata) == "Untitled Notes"


def test_build_citations_numbers_from_one_and_preserves_order():
    results = [
        ("id-1", 0.9, chunk_with_heading),
        ("id-2", 0.5, chunk_without_heading),
    ]
    citations = build_citations(results)

    assert citations == [
        Citation(source_index=1, label="week-09-day-01", heading="Type Hints", text="Type hints help readability."),
        Citation(source_index=2, label="Untitled Notes", heading=None, text="Napoleon lost at Waterloo."),
    ]


def test_build_citations_empty_results():
    assert build_citations([]) == []


def test_format_citations_shows_heading_when_present():
    citations = [Citation(source_index=1, label="week-09-day-01", heading="Type Hints", text="irrelevant")]
    assert format_citations(citations) == "[Source 01: week-09-day-01 - Type Hints]"


def test_format_citations_falls_back_when_heading_missing():
    citations = [Citation(source_index=2, label="Untitled Notes", heading=None, text="irrelevant")]
    assert format_citations(citations) == "[Source 02: Untitled Notes - No heading]"


def test_format_citations_joins_multiple_with_newline_no_trailing():
    citations = [
        Citation(source_index=1, label="week-09-day-01", heading="Type Hints", text="irrelevant"),
        Citation(source_index=2, label="Untitled Notes", heading=None, text="irrelevant"),
    ]
    expected = "[Source 01: week-09-day-01 - Type Hints]\n[Source 02: Untitled Notes - No heading]"
    assert format_citations(citations) == expected


def test_format_citations_empty_list():
    assert format_citations([]) == ""