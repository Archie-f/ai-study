import os
from pathlib import Path

from dotenv import load_dotenv


def get_notes_root() -> Path:
    """Load .env and read/validate the NOTES_ROOT environment variable.

    Returns:
        NOTES_ROOT as a Path.

    Raises:
        RuntimeError: if the NOTES_ROOT environment variable isn't set.
    """
    load_dotenv()
    path =  os.getenv("NOTES_ROOT")
    if path is None:
        raise RuntimeError("NOTES_ROOT environment variable doesn't exist  — check your .env file")
    return Path(path)

PROJECT_ROOT = Path(__file__).parent # rag-notes-qa/
PERSIST_PATH = str(PROJECT_ROOT / "persistent")
GOLDEN_QA_PATH = PROJECT_ROOT / "data" / "golden_qa.json"
EVAL_CORPUS_PATH = PROJECT_ROOT / "data" / "eval_corpus"
