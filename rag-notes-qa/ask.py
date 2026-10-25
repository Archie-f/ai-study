import argparse

from config import get_notes_root, PERSIST_PATH
from llm_compare.providers.openai_provider import OpenAIProvider
from rag_notes.adapters import ChromaVectorStore
from rag_notes.generate import answer_question
from rag_notes.pipeline import build_default_retriever


notes_root = get_notes_root()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--question",
    required=True
)
parser.add_argument(
    "--n",
    type=int,
    default=5
)
args = parser.parse_args()

vector_store = ChromaVectorStore(PERSIST_PATH)
retriever = build_default_retriever(notes_root, vector_store)
provider = OpenAIProvider()
answer_query = answer_question(
    retriever=retriever,
    question=args.question,
    provider=provider,
    n=args.n
)
print(answer_query.answer)