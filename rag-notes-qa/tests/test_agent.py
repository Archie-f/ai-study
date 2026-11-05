from types import SimpleNamespace

import pytest

from rag_notes.agent import parse_tool_uses, make_tool_result_message


def text_block(text="hello"):
    return SimpleNamespace(type="text", text=text)


def tool_block(tool_id, name="search_notes", **tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def test_empty_content_returns_empty_list():
    assert parse_tool_uses([]) == []


def test_text_only_returns_empty_list():
    assert parse_tool_uses([text_block()]) == []


def test_single_tool_use_is_returned_as_tuple():
    assert parse_tool_uses([tool_block("t1", query="RRF")]) == [
        ("t1", "search_notes", {"query": "RRF"})
    ]


def test_text_blocks_are_ignored_between_tool_uses():
    content = [text_block(), tool_block("t1", query="a"), text_block(), tool_block("t2", query="b", n=3)]
    assert parse_tool_uses(content) == [
        ("t1", "search_notes", {"query": "a"}),
        ("t2", "search_notes", {"query": "b", "n": 3}),
    ]


def test_order_is_preserved():
    ids = [t[0] for t in parse_tool_uses([tool_block("t3"), tool_block("t1"), tool_block("t2")])]
    assert ids == ["t3", "t1", "t2"]


def test_tool_name_is_passed_through():
    assert parse_tool_uses([tool_block("t1", name="other_tool")])[0][1] == "other_tool"


def test_empty_results_raise_value_error():
    with pytest.raises(ValueError):
        make_tool_result_message([])


def test_message_is_single_user_message():
    msg = make_tool_result_message([("t1", "ok", False)])
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)


def test_success_result_has_no_is_error_key():
    block = make_tool_result_message([("t1", "passage", False)])["content"][0]
    assert block == {"type": "tool_result", "tool_use_id": "t1", "content": "passage"}


def test_error_result_has_is_error_true():
    block = make_tool_result_message([("t1", "boom", True)])["content"][0]
    assert block == {"type": "tool_result", "tool_use_id": "t1", "content": "boom", "is_error": True}


def test_multiple_results_become_multiple_blocks_in_order():
    msg = make_tool_result_message([("t2", "b", False), ("t1", "a", True)])
    assert [b["tool_use_id"] for b in msg["content"]] == ["t2", "t1"]
    assert len(msg["content"]) == 2


def test_each_block_is_a_tool_result():
    msg = make_tool_result_message([("t1", "a", False), ("t2", "b", True)])
    assert all(b["type"] == "tool_result" for b in msg["content"])


# --- run_tool ---

from pathlib import Path
from typing import Literal

from rag_notes.agent import run_tool
from rag_notes.models import Chunk, DocumentMetadata

_metadata = DocumentMetadata(week=12, day=1, file_path=Path("fake.docx"), title="fake")
_chunk = Chunk(text="RRF merges rankings by position.", source=_metadata, heading="Hybrid Search", chunk_index=0)
_RESULTS = [("fake-0", 0.9, _chunk)]


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


def test_unknown_tool_is_error_and_does_not_search():
    r = FakeRetriever()
    content, is_error = run_tool("delete_all", {"query": "x"}, r)
    assert is_error is True
    assert "delete_all" in content
    assert r.calls == []


@pytest.mark.parametrize("tool_input", [{}, {"n": 3}, {"query": None}, {"query": 5}, {"query": ""}, {"query": "   "}])
def test_bad_query_is_error_and_does_not_search(tool_input):
    r = FakeRetriever()
    _, is_error = run_tool("search_notes", tool_input, r)
    assert is_error is True
    assert r.calls == []


def test_missing_n_defaults_to_five():
    r = FakeRetriever()
    run_tool("search_notes", {"query": "RRF"}, r)
    assert r.calls == [("RRF", 5, "hybrid")]


def test_input_dict_is_not_mutated():
    tool_input = {"query": "RRF"}
    run_tool("search_notes", tool_input, FakeRetriever())
    assert tool_input == {"query": "RRF"}


@pytest.mark.parametrize("n", [0, 11, -1, 2.5, "3", True, False, None])
def test_bad_n_is_error_and_does_not_search(n):
    r = FakeRetriever()
    _, is_error = run_tool("search_notes", {"query": "RRF", "n": n}, r)
    assert is_error is True
    assert r.calls == []


@pytest.mark.parametrize("n", [1, 10])
def test_n_boundaries_are_accepted_and_passed_through(n):
    r = FakeRetriever()
    _, is_error = run_tool("search_notes", {"query": "RRF", "n": n}, r)
    assert is_error is False
    assert r.calls == [("RRF", n, "hybrid")]


def test_empty_results_is_not_an_error():
    content, is_error = run_tool("search_notes", {"query": "zzz"}, FakeRetriever(results=[]))
    assert is_error is False
    assert content.startswith("No results")


def test_results_are_returned_as_built_context():
    content, is_error = run_tool("search_notes", {"query": "RRF"}, FakeRetriever())
    assert is_error is False
    assert "[Source 01:" in content
    assert "RRF merges rankings by position." in content


@pytest.mark.parametrize("error", [RuntimeError("db down"), KeyError("k"), ValueError("v")])
def test_retriever_exception_becomes_error_result(error):
    content, is_error = run_tool("search_notes", {"query": "RRF"}, FakeRetriever(error=error))
    assert is_error is True
    assert content


def test_query_is_passed_to_retriever_unchanged():
    r = FakeRetriever()
    run_tool("search_notes", {"query": "  RRF merge  "}, r)
    assert r.calls[0][0] == "  RRF merge  "


# --- run_agent ---

from rag_notes.agent import run_agent, AgentResult, AgentStepLimitError, SEARCH_NOTES_TOOL


def response(content, stop_reason, tokens_in=10, tokens_out=5):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


def tool_response(*blocks, **kw):
    return response(list(blocks), "tool_use", **kw)


def final_response(text="final answer", **kw):
    return response([text_block(text)], "end_turn", **kw)


class FakeMessages:
    """Same signature as client.messages.create. Replays queued responses and
    records a snapshot of every call (messages is copied: run_agent keeps mutating it)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, *, model, max_tokens, messages, tools):
        self.calls.append({"model": model, "max_tokens": max_tokens, "messages": list(messages), "tools": tools})
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def run(responses, retriever=None, **kw):
    client = FakeClient(responses)
    retriever = retriever or FakeRetriever()
    result = run_agent("What is RRF?", client, retriever, llm_model="fake-model", **kw)
    return result, client.messages.calls, retriever


def test_direct_answer_without_tools():
    result, calls, retriever = run([final_response("RRF merges rankings.", tokens_in=7, tokens_out=3)])
    assert result == AgentResult(answer="RRF merges rankings.", steps=1, tokens_in=7, tokens_out=3)
    assert len(calls) == 1
    assert retriever.calls == []


def test_answer_joins_all_text_blocks():
    r = response([text_block("Hello "), text_block("world")], "end_turn")
    result, _, _ = run([r])
    assert result.answer == "Hello world"


def test_answer_ignores_non_text_blocks():
    r = response([text_block("A"), tool_block("t1", query="x"), text_block("B")], "end_turn")
    result, _, _ = run([r])
    assert result.answer == "AB"


def test_model_max_tokens_and_tools_are_passed_on_every_call():
    _, calls, _ = run([tool_response(tool_block("t1", query="RRF")), final_response()], max_tokens=321)
    assert len(calls) == 2
    for call in calls:
        assert call["model"] == "fake-model"
        assert call["max_tokens"] == 321
        assert call["tools"] == [SEARCH_NOTES_TOOL]


def test_first_call_contains_only_the_question():
    _, calls, _ = run([final_response()])
    assert calls[0]["messages"] == [{"role": "user", "content": "What is RRF?"}]


def test_tool_round_then_answer_counts_steps_and_sums_tokens():
    responses = [
        tool_response(tool_block("t1", query="RRF"), tokens_in=100, tokens_out=20),
        final_response("done", tokens_in=300, tokens_out=40),
    ]
    result, calls, retriever = run(responses)
    assert result.answer == "done"
    assert result.steps == 2
    assert result.tokens_in == 400
    assert result.tokens_out == 60
    assert retriever.calls == [("RRF", 5, "hybrid")]


def test_second_call_has_alternating_roles_and_unchanged_assistant_content():
    first = tool_response(tool_block("t1", query="RRF"))
    _, calls, _ = run([first, final_response()])
    msgs = calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] is first.content
    block = msgs[2]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "t1"
    assert "RRF merges rankings by position." in block["content"]


def test_two_tools_in_one_response_give_one_user_message_with_two_results_in_order():
    first = tool_response(tool_block("t1", query="a"), tool_block("t2", query="b"))
    _, calls, retriever = run([first, final_response()])
    msgs = calls[1]["messages"]
    assert len(msgs) == 3
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["t1", "t2"]
    assert [c[0] for c in retriever.calls] == ["a", "b"]


def test_unknown_tool_becomes_error_result_and_loop_continues():
    first = tool_response(tool_block("t1", name="delete_all"))
    result, calls, retriever = run([first, final_response("recovered")])
    block = calls[1]["messages"][2]["content"][0]
    assert block["is_error"] is True
    assert result.answer == "recovered"
    assert retriever.calls == []


def test_retriever_exception_becomes_error_result_and_loop_continues():
    first = tool_response(tool_block("t1", query="RRF"))
    result, calls, _ = run([first, final_response("recovered")], retriever=FakeRetriever(error=KeyError("k")))
    assert calls[1]["messages"][2]["content"][0]["is_error"] is True
    assert result.answer == "recovered"


def test_empty_search_is_not_an_error_result():
    first = tool_response(tool_block("t1", query="zzz"))
    _, calls, _ = run([first, final_response()], retriever=FakeRetriever(results=[]))
    block = calls[1]["messages"][2]["content"][0]
    assert "is_error" not in block
    assert block["content"].startswith("No results")


def test_step_limit_raises_after_exactly_max_steps_calls():
    responses = [tool_response(tool_block(f"t{i}", query="RRF")) for i in range(10)]
    client = FakeClient(responses)
    with pytest.raises(AgentStepLimitError):
        run_agent("q", client, FakeRetriever(), llm_model="m", max_steps=3)
    assert len(client.messages.calls) == 3


def test_answer_on_the_last_allowed_step_is_not_an_error():
    responses = [tool_response(tool_block("t1", query="RRF")), final_response("ok")]
    result, _, _ = run(responses, max_steps=2)
    assert result.steps == 2
    assert result.answer == "ok"


def test_max_steps_one_with_tool_request_raises():
    with pytest.raises(AgentStepLimitError):
        run([tool_response(tool_block("t1", query="RRF"))], max_steps=1)


def test_no_tool_call_gives_empty_tool_calls():
    result, _, _ = run([final_response()])
    assert result.tool_calls == []


def test_tool_calls_record_name_and_input_without_id():
    first = tool_response(tool_block("t1", query="RRF", n=3))
    result, _, _ = run([first, final_response()])
    assert result.tool_calls == [("search_notes", {"query": "RRF", "n": 3})]


def test_tool_calls_of_all_rounds_are_kept_in_order():
    responses = [
        tool_response(tool_block("t1", query="a")),
        tool_response(tool_block("t2", query="b")),
        final_response(),
    ]
    result, _, _ = run(responses)
    assert [c[1]["query"] for c in result.tool_calls] == ["a", "b"]


def test_two_tools_in_one_response_keep_their_order():
    first = tool_response(tool_block("t1", query="a"), tool_block("t2", query="b"))
    result, _, _ = run([first, final_response()])
    assert [c[1]["query"] for c in result.tool_calls] == ["a", "b"]


def test_failed_tool_call_is_still_recorded():
    first = tool_response(tool_block("t1", name="delete_all"))
    result, _, _ = run([first, final_response()])
    assert result.tool_calls == [("delete_all", {})]


def test_each_run_gets_its_own_tool_calls_list():
    r1, _, _ = run([tool_response(tool_block("t1", query="a")), final_response()])
    r2, _, _ = run([final_response()])
    assert r1.tool_calls != r2.tool_calls
    assert r2.tool_calls == []


def test_custom_tool_is_sent_on_every_call_instead_of_the_default():
    custom = {**SEARCH_NOTES_TOOL, "description": "other description"}
    _, calls, _ = run([tool_response(tool_block("t1", query="RRF")), final_response()], tool=custom)
    assert len(calls) == 2
    for call in calls:
        assert call["tools"] == [custom]
        assert call["tools"][0]["description"] == "other description"


def test_tool_none_uses_search_notes_tool():
    _, calls, _ = run([final_response()], tool=None)
    assert calls[0]["tools"] == [SEARCH_NOTES_TOOL]


def test_custom_tool_does_not_change_the_default_tool():
    before = dict(SEARCH_NOTES_TOOL)
    run([final_response()], tool={**SEARCH_NOTES_TOOL, "description": "x"})
    assert SEARCH_NOTES_TOOL == before
    _, calls, _ = run([final_response()])
    assert calls[0]["tools"] == [SEARCH_NOTES_TOOL]


def test_end_turn_gives_stop_reason_end_turn():
    result, _, _ = run([final_response()])
    assert result.stop_reason == "end_turn"


def test_max_tokens_keeps_the_partial_text_and_reports_the_reason():
    r = response([text_block("The answer is cut o")], "max_tokens")
    result, _, _ = run([r])
    assert result.answer == "The answer is cut o"
    assert result.stop_reason == "max_tokens"


def test_refusal_is_returned_not_raised():
    r = response([text_block("I can't help with that.")], "refusal")
    result, _, _ = run([r])
    assert result.stop_reason == "refusal"
    assert result.answer == "I can't help with that."


@pytest.mark.parametrize("reason", ["stop_sequence", "pause_turn"])
def test_other_stop_reasons_are_passed_through(reason):
    result, _, _ = run([response([text_block("x")], reason)])
    assert result.stop_reason == reason


def test_stop_reason_is_the_one_of_the_last_response_after_tool_rounds():
    responses = [tool_response(tool_block("t1", query="RRF")), response([text_block("cut")], "max_tokens")]
    result, _, _ = run(responses)
    assert result.steps == 2
    assert result.stop_reason == "max_tokens"


def test_tool_use_responses_do_not_leak_into_stop_reason():
    responses = [tool_response(tool_block("t1", query="RRF")), final_response()]
    result, _, _ = run(responses)
    assert result.stop_reason == "end_turn"
