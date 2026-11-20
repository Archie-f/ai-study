from langchain_core.messages import AIMessage

from day02_exercise01_read_message import requested_tools


def test_requested_tools():
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_weather", "args": {"city": "Oslo"},   "id": "w1"},
            {"name": "get_weather", "args": {"city": "Bergen"}, "id": "w2"}]
    )
    assert requested_tools(message) == [("get_weather", {"city": "Oslo"}), ("get_weather", {"city": "Bergen"})]


def test_requested_tools_with_empty_tool_calls():
    message = AIMessage(
        content="It is sunny."
    )
    assert requested_tools(message) == []