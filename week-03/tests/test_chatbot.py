import pytest

from chatbot.chatbot import add_message

@pytest.fixture
def empty_history():
    return []

@pytest.fixture
def system_message():
    return [{"role": "system", "content": "You are a professional assistant"}]

def test_add_message_appends_to_history(empty_history):
    add_message(empty_history, "user", "hello")
    assert len(empty_history) == 1
    assert empty_history[0]["role"] == "user"
    assert empty_history[0]["content"] == "hello"

def test_add_message_preserves_existing_history(system_message):
    add_message(system_message, "user", "hello")
    assert len(system_message) == 2
    assert system_message[0]["role"] == "system"

def test_add_message_assistant_role(system_message):
    add_message(system_message, "assistant", "How can I help you?")
    assert len(system_message) == 2
    assert system_message[1]["role"] == "assistant"
