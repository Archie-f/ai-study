"""Tests for rag_notes.agent_langgraph with a fake chat model and a fake retriever (no API calls)."""

import pytest
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from rag_notes.agent import AgentStepLimitError, AgentResult, SEARCH_NOTES_TOOL
from rag_notes.agent_langgraph import route_after_model, run_langgraph_agent
from rag_notes.models import Chunk, DocumentMetadata


_metadata = DocumentMetadata(week=12, day=1, file_path=Path("fake.docx"), title="fake")
_chunk = Chunk(text="RRF merges rankings by position.", source=_metadata, heading="Hybrid Search", chunk_index=0)
_RESULTS = [("fake-0", 0.9, _chunk)]


class FakeChatModel:
    """Plays back a fixed list of AIMessages, one per invoke. Records everything."""

    def __init__(self, replies: list[AIMessage]):
        """Copy the replies so pop() does not empty the caller's list."""
        self.replies = list(replies)
        self.bound_tools = None      # what bind_tools received
        self.received = []           # the messages of every invoke, in order

    def bind_tools(self, tools):
        """Record the tools and return self, like ChatAnthropic returns a bound model."""
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        """Record a copy of the conversation and return the next scripted reply."""
        self.received.append(list(messages))
        return self.replies.pop(0)


class FakeRetriever:
    """Stand-in Retriever: returns fixed results (or raises), records every call."""

    def __init__(self, results=None, error: Exception | None = None):
        self.results = _RESULTS if results is None else results
        self.error = error
        self.calls = []

    def search(self, query: str, n: int = 3, mode: Literal["hybrid", "vector", "bm25"] = "hybrid"):
        self.calls.append((query, n, mode))
        if self.error:
            raise self.error
        return self.results


def ai_tool_call(call_id="t1", query="RRF", tokens=(900, 40)):
    """An AIMessage asking for one search_notes call, shaped like ChatAnthropic's."""
    return AIMessage(
        content="",
        tool_calls=[{"name": "search_notes", "args": {"query": query}, "id": call_id}],
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                        "total_tokens": sum(tokens)},
        response_metadata={"stop_reason": "tool_use"},
    )

def ai_answer(text="RRF is ...", tokens=(2100, 150), stop_reason="end_turn"):
    """A final AIMessage with text and no tool call, shaped like ChatAnthropic's."""
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                        "total_tokens": sum(tokens)},
        response_metadata={"stop_reason": stop_reason},
    )


def test_router_returns_tools_when_last_message_asks_for_a_tool():
    """A last message with a tool call routes to the tools node."""
    assert route_after_model({"messages": [HumanMessage("q"), ai_tool_call()]}) == "tools"


def test_router_returns_end_for_a_plain_answer():
    """A last message without tool calls ends the graph."""
    assert route_after_model({"messages": [HumanMessage("q"), ai_answer()]}) == "end"


def test_tool_error_reaches_the_model_with_status_error():
    """An empty query is never searched and comes back to the model as status="error"."""
    retriever = FakeRetriever()
    llm = FakeChatModel([ai_tool_call(query=""), ai_answer()])
    run_langgraph_agent("q", llm, retriever)
    assert [(msg.tool_call_id, msg.status)
            for msg in llm.received[1] if isinstance(msg, ToolMessage)] == [('t1', 'error')]
    assert retriever.calls == []


def test_step_limit_raises_agent_step_limit_error():
    """max_steps=3 stops after exactly 3 model calls with AgentStepLimitError."""
    llm = FakeChatModel([ai_tool_call(call_id='t1'), ai_tool_call(call_id='t2'), ai_tool_call(call_id='t3')])
    with pytest.raises(AgentStepLimitError, match="Model still requested tools after 3 steps"):
        run_langgraph_agent("q", llm, FakeRetriever(), max_steps=3)
    assert len(llm.received) == 3


def test_one_search_then_answer_returns_the_full_result():
    """Answer, steps, summed tokens, flat tool_calls and stop_reason are read from all AIMessages."""
    llm = FakeChatModel([ai_tool_call(), ai_answer()])
    agent_result = run_langgraph_agent("q", llm, FakeRetriever())
    assert agent_result == AgentResult(answer="RRF is ...", steps=2, tokens_in=3000, tokens_out=190,
                    tool_calls=[("search_notes", {"query": "RRF"})], stop_reason="end_turn")


def test_two_tool_calls_in_one_message_get_two_tool_messages_in_order():
    """Every tool call of one message gets its own ToolMessage, in order, and its own search."""
    two_tool_calls = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_notes", "args": {"query": "RRF"}, "id": "t1"},
            {"name": "search_notes", "args": {"query": "BM25"}, "id": "t2"},
        ],
        usage_metadata={"input_tokens": 900, "output_tokens": 40, "total_tokens": 940},
        response_metadata={"stop_reason": "tool_use"},
    )
    retriever = FakeRetriever()
    llm = FakeChatModel([two_tool_calls, ai_answer()])
    run_langgraph_agent("q", llm, retriever)
    tool_call_ids = [msg.tool_call_id for msg in llm.received[1] if isinstance(msg, ToolMessage)]
    assert tool_call_ids == ["t1", "t2"]
    assert len(retriever.calls) == 2


def test_plain_answer_never_runs_the_tool():
    """A direct answer is one model call, no tool calls and no search."""
    retriever = FakeRetriever()
    llm = FakeChatModel([ai_answer()])
    agent_result = run_langgraph_agent("q", llm, retriever)
    assert agent_result.steps == 1
    assert agent_result.tool_calls == []
    assert retriever.calls == []


def test_given_tool_is_bound_as_a_one_element_list():
    """The tool argument is what gets bound, inside a list."""
    other = {**SEARCH_NOTES_TOOL, "description": "other"}
    llm = FakeChatModel([ai_tool_call(), ai_answer()])
    run_langgraph_agent("q", llm, FakeRetriever(), tool=other)
    assert llm.bound_tools == [other]


def test_no_tool_binds_search_notes_tool():
    """Without a tool argument, SEARCH_NOTES_TOOL is bound."""
    llm = FakeChatModel([ai_tool_call(), ai_answer()])
    run_langgraph_agent("q", llm, FakeRetriever())
    assert llm.bound_tools == [SEARCH_NOTES_TOOL]


def test_stop_reason_of_the_last_call_is_returned():
    """The stop_reason comes from the last model call, not the first."""
    llm = FakeChatModel([ai_tool_call(), ai_answer(stop_reason="max_tokens")])
    agent_result = run_langgraph_agent("q", llm, FakeRetriever())
    assert agent_result.stop_reason == "max_tokens"


def test_default_max_steps_allows_five_model_calls_and_five_searches():
    """The default limit stops where run_agent does: 5 model calls, 5 searches."""
    retriever = FakeRetriever()
    llm = FakeChatModel([ai_tool_call(), ai_tool_call(), ai_tool_call(), ai_tool_call(), ai_tool_call()])
    with pytest.raises(AgentStepLimitError, match="Model still requested tools after 5 steps"):
        run_langgraph_agent("q", llm, retriever)
    assert len(llm.received) == 5
    assert len(retriever.calls) == 5


def test_step_limit_error_is_raised_from_graph_recursion_error():
    """The step-limit error keeps LangGraph's GraphRecursionError as its __cause__."""
    llm = FakeChatModel([ai_tool_call(), ai_tool_call(), ai_tool_call()])
    with pytest.raises(AgentStepLimitError, match="Model still requested tools after 3 steps") as exc_info:
        run_langgraph_agent("q", llm, FakeRetriever(), max_steps=3)
    assert isinstance(exc_info.value.__cause__, GraphRecursionError)
