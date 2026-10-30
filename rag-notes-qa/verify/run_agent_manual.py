import os

import anthropic
from dotenv import load_dotenv

from config import PERSIST_PATH, EVAL_CORPUS_PATH
from rag_notes.adapters import ChromaVectorStore
from rag_notes.agent import run_agent, AgentStepLimitError
from rag_notes.pipeline import build_default_retriever

load_dotenv()

client = anthropic.Anthropic()
llm_model = os.environ["ANTHROPIC_MODEL_NAME"]
QUESTIONS = [
    ("lexical", "How does Reciprocal Rank Fusion score a chunk?"),                                  # q038
    ("paraphrase", "Why set a time limit on requests to a model that runs on your own laptop?"),    # q024
    ("unanswerable", "What is the capital of Norway?"),
]


def main() -> None:
    vector_store = ChromaVectorStore(PERSIST_PATH)
    retriever = build_default_retriever(EVAL_CORPUS_PATH, vector_store)

    results = []
    for q_type, question in QUESTIONS:
        try:
            agent_result = run_agent(question, client, retriever, llm_model)
            results.append(
                {
                q_type:
                    {
                    "answer": agent_result.answer,
                    "steps": agent_result.steps,
                    "tokens_in": agent_result.tokens_in,
                    "tokens_out": agent_result.tokens_out
                    }
                }
            )
        except AgentStepLimitError as e:
            results.append(f"{q_type}: {e}")

    print()
    print(f"{'type':<14}{'steps':>6}{'tokens_in':>11}{'tokens_out':>12}{'tool':>6}")
    print("-" * 49)
    for item in results:
        if isinstance(item, str):
            print(item)
            continue
        for q_type, r in item.items():
            tool_called = "yes" if r["steps"] > 1 else "no"
            print(f"{q_type:<14}{r['steps']:>6}{r['tokens_in']:>11}{r['tokens_out']:>12}{tool_called:>6}")

    for item in results:
        if isinstance(item, str):
            continue
        for q_type, r in item.items():
            print(f"\n[{q_type}]\n{r['answer']}")


if __name__ == "__main__":
    main()
