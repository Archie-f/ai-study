from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError


class ShrinkState(TypedDict):
    value: float
    steps: int

def halve(state: ShrinkState) -> dict:
    """Halve value and count one step.

    Example: {"value": 100.0, "steps": 0} -> {"value": 50.0, "steps": 1}
    """
    val = state["value"] / 2
    steps = state["steps"] + 1
    return {"value": val, "steps": steps}

def should_continue(state: ShrinkState) -> str:
    """Router: "again" while value is above 10.0, otherwise "done".

    Examples: value 12.5 -> "again"; value 6.25 -> "done"; value 10.0 -> "done"
    """
    return "again" if state["value"] > 10.0 else "done"

def build_shrink_graph():
    """Compile: START -> halve -> (router) -> halve again, or END."""
    builder = StateGraph(ShrinkState)               # type: ignore
    builder.add_node("halve", halve)                # type: ignore
    builder.add_edge(START, "halve")
    builder.add_conditional_edges("halve", should_continue, {"again": "halve", "done": END})
    return builder.compile()


if __name__ == "__main__":
    values = [100.0, 8.0, 1e9]

    graph = build_shrink_graph()
    results = []
    for value in values:
        try:
            result = graph.invoke({"value": value, "steps": 0}, config={"recursion_limit": 5})
            results.append(result)
        except GraphRecursionError as e:
            print(type(e).__name__)

    print(results)
