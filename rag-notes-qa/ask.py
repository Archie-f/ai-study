import argparse
from pathlib import Path

from config import get_notes_root
from llm_compare.providers.ollama_provider import OllamaProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from rag_notes.adapters import StructureChunker, HybridRetriever
from rag_notes.generate import answer_question
from rag_notes.retrieval import build_retrieval_index


PERSIST_PATH = str(Path(__file__).parent / "persistent")

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

index = build_retrieval_index(notes_root, PERSIST_PATH, StructureChunker())
retriever = HybridRetriever(index)
provider = OpenAIProvider()
answer_query = answer_question(
    retriever=retriever,
    question=args.question,
    provider=provider,
    n=args.n
)
print(answer_query.answer)