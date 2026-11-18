from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from rag_notes.agent_langgraph import build_agent_graph


class AlwaysToolModel:
    """Fake model: every call asks for one search. Counts its calls."""
    def __init__(self):
        self.calls = 0
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[{"name": "search_notes", "args": {"query": "q"}, "id": f"t{self.calls}"}],
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            response_metadata={"stop_reason": "tool_use"},
        )


class CountingRetriever:
    def __init__(self):
        self.runs = 0
    def search(self, query, n):
        self.runs += 1
        return ["chunk"]


if "__main__" == __name__:
    limits = [5, 9, 10, 11]

    for limit in limits:
        model = AlwaysToolModel()
        retriever = CountingRetriever()
        graph = build_agent_graph(model, retriever)         # type: ignore

        try:
            graph.invoke(
                {"messages": [HumanMessage("What is RRF?")]},
                config={"recursion_limit": limit}
            )
        except GraphRecursionError as err:
            print(f"Model is called '{model.calls}' times."
            f"Retriever ran '{retriever.runs}' times.")
