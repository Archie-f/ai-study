import copy
from pathlib import Path

import pytest

from rag_notes.golden_qa import validate_golden_entries
from rag_notes.models import Chunk, DocumentMetadata


def make_meta(title: str) -> DocumentMetadata:
    return DocumentMetadata(week=None, day=None, file_path=Path(f"{title}.docx"), title=title)


CHUNKS = [
    Chunk(text="Type hints let you\nannotate your code.\n", source=make_meta("w01"), heading=None, chunk_index=1),
    Chunk(text="build_messages() is the bridge between them", source=make_meta("w05"), heading=None, chunk_index=3),
]

VALID = [
    {"id": "q001", "question": "Q?", "source_document": "w01",
     "expected_answer_text": "... Type hints let you annotate your code ...",
     "answer_key": "Type hints let you annotate your code", "notes": ""},
    {"id": "q002", "question": "Q?", "source_document": "w05",
     "expected_answer_text": "build_messages() is the bridge between them.",
     "answer_key": "build_messages() is the bridge between them", "notes": "paraphrase"},
]

LONG_KEY = " ".join(["word"] * 31)


@pytest.fixture
def entries():
    return copy.deepcopy(VALID)


def test_valid_entries_give_no_problems(entries):
    assert validate_golden_entries(entries, CHUNKS) == []


def test_optional_notes_may_be_missing(entries):
    del entries[1]["notes"]
    assert validate_golden_entries(entries, CHUNKS) == []


def test_unknown_empty_field_is_ignored(entries):
    entries[1]["extra"] = ""
    assert validate_golden_entries(entries, CHUNKS) == []


@pytest.mark.parametrize("field", ["question", "source_document", "expected_answer_text", "answer_key"])
def test_absent_required_field_is_reported(entries, field):
    del entries[1][field]
    assert validate_golden_entries(entries, CHUNKS) == [f"q002: missing {field}"]


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_or_blank_answer_key_is_reported_as_missing(entries, value):
    entries[1]["answer_key"] = value
    assert validate_golden_entries(entries, CHUNKS) == ["q002: missing answer_key"]


@pytest.mark.parametrize("value", [None, ""])
def test_missing_id_uses_placeholder(entries, value):
    if value is None:
        del entries[1]["id"]
    else:
        entries[1]["id"] = value
    assert validate_golden_entries(entries, CHUNKS) == ["<no id>: missing id"]


def test_several_missing_fields_in_one_message_and_no_further_checks(entries):
    entries[1]["question"] = ""
    entries[1]["source_document"] = ""
    assert validate_golden_entries(entries, CHUNKS) == ["q002: missing question, source_document"]


def test_duplicate_id_reported_once_for_the_second_entry(entries):
    entries[1]["id"] = "q001"
    assert validate_golden_entries(entries, CHUNKS) == ["q001: duplicate id"]


def test_first_of_duplicates_is_still_fully_checked(entries):
    entries[1]["id"] = "q001"
    entries[0]["answer_key"] = "kubernetes helm"
    entries[0]["expected_answer_text"] = "kubernetes helm"
    assert validate_golden_entries(entries, CHUNKS) == ["q001: answer_key matches no chunk", "q001: duplicate id"]


def test_same_id_three_times_gives_two_messages(entries):
    entries[1]["id"] = "q001"
    entries.append(copy.deepcopy(entries[0]))
    assert validate_golden_entries(entries, CHUNKS) == ["q001: duplicate id", "q001: duplicate id"]


def test_key_not_inside_expected_answer_text(entries):
    entries[1]["expected_answer_text"] = "Something else."
    assert validate_golden_entries(entries, CHUNKS) == ["q002: answer_key is not inside expected_answer_text"]


def test_key_inside_expected_text_after_normalization(entries):
    entries[1]["answer_key"] = "BUILD_MESSAGES() is   the bridge\nbetween them"
    assert validate_golden_entries(entries, CHUNKS) == []


def test_key_longer_than_limit(entries):
    entries[1]["answer_key"] = LONG_KEY
    entries[1]["expected_answer_text"] = LONG_KEY
    assert validate_golden_entries(entries, CHUNKS) == [
        "q002: answer_key has 31 words (max 30)",
        "q002: answer_key matches no chunk",
    ]


def test_key_at_limit_is_accepted(entries):
    assert validate_golden_entries(entries, CHUNKS, max_key_words=7) == []


def test_key_in_no_chunk(entries):
    entries[1]["answer_key"] = "kubernetes helm"
    entries[1]["expected_answer_text"] = "kubernetes helm"
    assert validate_golden_entries(entries, CHUNKS) == ["q002: answer_key matches no chunk"]


def test_key_only_in_another_document(entries):
    entries[1]["source_document"] = "w01"
    assert validate_golden_entries(entries, CHUNKS) == ["q002: answer_key matches no chunk in w01"]
