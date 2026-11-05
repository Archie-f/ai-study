import json
import os
from itertools import chain

import anthropic
from dotenv import load_dotenv

from config import GOLDEN_QA_PATH, PERSIST_PATH, EVAL_CORPUS_PATH
from rag_notes.adapters import ChromaVectorStore
from rag_notes.agent import SEARCH_NOTES_TOOL, run_agent, AgentStepLimitError
from rag_notes.pipeline import build_default_retriever


OLD_DESCRIPTION = ("Search the user's daily takeaway notes and return the passages that are the most "
                   "relevant to the question. Use it for any question about what the notes say. Do not "
                   "use it for any general knowledge questions that the notes do not cover.")
CONTROL_QUESTIONS = [
    "What is the capital of Norway?",
    "How many minutes should I boil an egg for a soft yolk?",
    "Who wrote the novel '1984'?",
    "What is the boiling point of water in Celsius at sea level?",
    "Which planet is the largest in our solar system?",
    "How do I convert 5 miles to kilometres?"
]
OLD_TOOL = {**SEARCH_NOTES_TOOL, "description": OLD_DESCRIPTION}


def tool_call_rates(rows: list[tuple[str, bool]]) -> dict[str, tuple[int, int]]:
    """Return {group: (questions that called a tool, questions in the group)}.

    rows: (group, tool_called) per question. Groups appear in the order first seen.
    No rows -> {}.
    """
    result = {}

    for group, is_called in rows:
        if group in result:
            continue

        groups = [row for row in rows if row[0] == group]
        called = sum(is_called for group, is_called in groups)
        result[group] = (called, len(groups))

    return result


def main():
    load_dotenv()
    client = anthropic.Anthropic()
    llm_model = os.environ["ANTHROPIC_MODEL_NAME"]
    vector_store = ChromaVectorStore(PERSIST_PATH)
    retriever = build_default_retriever(EVAL_CORPUS_PATH, vector_store)

    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa_list = json.load(f)

    n_quests = []
    for entry in golden_qa_list:
        n_quests.append(("notes", entry["question"]))

    c_quests = []
    for c_quest in CONTROL_QUESTIONS:
        c_quests.append(("controls", c_quest))

    tools = [("new", None)]
    for label, tool in tools:
        print(f"Verifying {label}:")
        rows = []
        hit_step_limit = {}
        missed = []
        false_alarms = []
        for group, question in chain(n_quests, c_quests):
            try:
                agent_result = run_agent(
                    question=question,
                    client=client,
                    retriever=retriever,
                    llm_model=llm_model,
                    tool=tool
                )
                tool_called = len(agent_result.tool_calls) > 0
                rows.append((group, tool_called))

                if group == "notes" and not tool_called:
                    missed.append(question)

                if group == "controls" and tool_called:
                    false_alarms.append(question)

            except AgentStepLimitError:
                rows.append((group, True))
                hit_step_limit[group] = hit_step_limit.get(group, 0) + 1
                print(f"Hit step limit:: {question}")

        call_rates = tool_call_rates(rows)

        for group, (called, total) in call_rates.items():
            main_text = f"{label} / {group}: {called}/{total} called "
            skipped_text = f"({hit_step_limit.get(group)} hit step limit)" if hit_step_limit.get(group) is not None else ""
            print(f"{main_text} {skipped_text}")

        print("\nMissed Questions:")
        print("\n".join(missed))

        print("\nFalse Alarms:")
        print("\n".join(false_alarms))
        print()


if __name__ == "__main__":
    main()
