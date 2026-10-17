"""Hand-made test data for validate_golden_entries() (Week 16 Day 02).

Run from rag-notes-qa/ with the venv active:
    python ../week-16/day02_validator_cases.py

Each case prints the actual result next to the expected one.
"""
import copy
from pathlib import Path

from rag_notes.models import Chunk, DocumentMetadata
from rag_notes.golden_qa import validate_golden_entries


def make_meta(title: str) -> DocumentMetadata:
    return DocumentMetadata(week=None, day=None, file_path=Path(f"{title}.docx"), title=title)


CHUNKS = [
    Chunk("Type hints let you\nannotate your code.\n", make_meta("w01"), None, 1),
    Chunk("build_messages() is the bridge between them", make_meta("w05"), None, 3),
]

VALID = [
    {"id": "q001", "question": "Q?", "source_document": "w01",
     "expected_answer_text": "... Type hints let you annotate your code ...",
     "answer_key": "Type hints let you annotate your code", "notes": ""},
    {"id": "q002", "question": "Q?", "source_document": "w05",
     "expected_answer_text": "build_messages() is the bridge between them.",
     "answer_key": "build_messages() is the bridge between them", "notes": "paraphrase"},
]


def case(label, change, expected):
    entries = copy.deepcopy(VALID)
    change(entries)
    try:
        actual = validate_golden_entries(entries, CHUNKS)
    except Exception as error:
        actual = f"{type(error).__name__}: {error}"
    status = "OK  " if actual == expected else "FAIL"
    print(f"{status} {label}\n     actual:   {actual}\n     expected: {expected}\n")


def remove(field, index=1):
    return lambda entries: entries[index].pop(field)


def set_value(field, value, index=1):
    return lambda entries: entries[index].__setitem__(field, value)


# --- check 1: required fields -------------------------------------------------
case("two valid entries", lambda e: None, [])
case("notes missing (optional)", remove("notes"), [])
case("question empty", set_value("question", ""), ["q002: missing question"])
case("question absent", remove("question"), ["q002: missing question"])
case("answer_key absent", remove("answer_key"), ["q002: missing answer_key"])
case("answer_key only spaces", set_value("answer_key", "   "), ["q002: missing answer_key"])
case("id absent", remove("id"), ["<no id>: missing id"])
case("id empty", set_value("id", ""), ["<no id>: missing id"])
case("two fields empty", lambda e: (e[1].__setitem__("question", ""), e[1].__setitem__("source_document", "")),
     ["q002: missing question, source_document"])
case("unknown extra field empty", set_value("extra", ""), [])

# --- check 2: duplicate id ----------------------------------------------------
case("same id twice", set_value("id", "q001"), ["q001: duplicate id"])

# --- check 3: key inside expected_answer_text ---------------------------------
case("key not in expected text", set_value("expected_answer_text", "Something else."),
     ["q002: answer_key is not inside expected_answer_text"])

# --- check 4: key length ------------------------------------------------------
LONG = " ".join(["word"] * 31)
case("31-word key", lambda e: (e[1].__setitem__("answer_key", LONG), e[1].__setitem__("expected_answer_text", LONG)),
     ["q002: answer_key has 31 words (max 30)", "q002: answer_key matches no chunk"])

# --- check 5: key matches a chunk ---------------------------------------------
case("key in no chunk", lambda e: (e[1].__setitem__("answer_key", "kubernetes helm"),
                                   e[1].__setitem__("expected_answer_text", "kubernetes helm")),
     ["q002: answer_key matches no chunk"])

# --- check 6: match in source_document ----------------------------------------
case("key only in another document", set_value("source_document", "w01"),
     ["q002: answer_key matches no chunk in w01"])


# --- check 2, added after review: the first occurrence must still be checked ---
def duplicate_and_bad_first_key(entries):
    entries[1]["id"] = "q001"
    entries[0]["answer_key"] = "kubernetes helm"
    entries[0]["expected_answer_text"] = "kubernetes helm"


case("duplicate + first entry has a bad key", duplicate_and_bad_first_key,
     ["q001: answer_key matches no chunk", "q001: duplicate id"])
case("same id three times", lambda e: (e[1].__setitem__("id", "q001"), e.append(copy.deepcopy(e[0]))),
     ["q001: duplicate id", "q001: duplicate id"])
