from pathlib import Path

from docx import Document

from rag_notes.loader import load_corpus


def test_load_corpus_returns_documents_in_alphabetical_order(tmp_path, monkeypatch):
    """load_corpus must not depend on file system order: even if rglob returns reversed paths, documents come back sorted."""
    for day in (1, 2, 3):
        Document().save(str(tmp_path / f"week-01-day-0{day}-takeaway-notes.docx"))

    original_rglob = Path.rglob

    def reversed_rglob(self, pattern):
        return iter(sorted(original_rglob(self, pattern), reverse=True))

    monkeypatch.setattr(Path, "rglob", reversed_rglob)

    titles = [document.metadata.title for document in load_corpus(tmp_path)]

    assert titles == [
        "week-01-day-01-takeaway-notes",
        "week-01-day-02-takeaway-notes",
        "week-01-day-03-takeaway-notes",
    ]
