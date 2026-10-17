from rag_notes.models import Chunk
from rag_notes.retrieval_metrics import normalize_text, find_relevant_chunk_ids


REQUIRED_FIELDS = ("id", "question", "source_document", "expected_answer_text", "answer_key")


def validate_golden_entries(
    entries: list[dict],
    chunks: list[Chunk],
    max_key_words: int = 30,
) -> list[str]:
    """Check every golden entry and return one message per problem found.

    Checks, in order, per entry: required fields present and non-empty;
    id not seen before; normalized answer_key inside normalized
    expected_answer_text; answer_key at most max_key_words words;
    answer_key matches at least one chunk; at least one match is in
    the entry's source_document.

    Args:
        entries: the parsed golden_qa.json list
        chunks: every chunk of the evaluation index
        max_key_words: upper limit for answer_key length in words
    Returns:
        problem messages, each starting with "<id>: "; empty if all valid
    """
    messages = []
    seen_ids = set()
    for entry in entries:
        #1. id, question, source_document, expected_answer_text, answer_key exist and not empty
        non_existing_fields = [field for field in REQUIRED_FIELDS if not entry.get(field) or entry.get(field).strip() == ""]
        entry_id = "<no id>" if not entry.get("id") else entry.get("id")

        if non_existing_fields:
            message = f"{entry_id}: missing {", ".join(non_existing_fields)}"
            messages.append(message)
            continue

        #2. id is not duplicate
        if entry_id in seen_ids:
            message = f"{entry_id}: duplicate id"
            messages.append(message)
        seen_ids.add(entry_id)

        #3. normalized answer_key is in normalized expected_answer_text
        if entry.get("answer_key"):
            answer_key = normalize_text(entry["answer_key"])
            expected_answer_text = normalize_text(entry["expected_answer_text"])
            if answer_key not in expected_answer_text:
                message = f"{entry['id']}: answer_key is not inside expected_answer_text"
                messages.append(message)

        #4. answer_key has no more word count than max_key_words
            word_count = len(answer_key.split())
            if  word_count > max_key_words:
                message = f"{entry_id}: answer_key has {word_count} words (max {max_key_words})"
                messages.append(message)

        #5. answer_key exists at least in one chunk
            if answer_key != "":
                relevant_chunk_ids = find_relevant_chunk_ids(chunks, answer_key)
                if len(relevant_chunk_ids) == 0:
                    message = f"{entry_id}: answer_key matches no chunk"
                    messages.append(message)

        #6. answer_key matches at least one chunk in source_document
                else:
                    relevant_chunks = [chunk for chunk in chunks if chunk.source.title == entry["source_document"]]
                    chunk_ids = find_relevant_chunk_ids(relevant_chunks, answer_key)
                    if len(chunk_ids) == 0:
                        message = f"{entry_id}: answer_key matches no chunk in {entry['source_document']}"
                        messages.append(message)

    return messages
