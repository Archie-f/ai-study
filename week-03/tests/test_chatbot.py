import pytest

from unittest.mock import patch
from chatbot.chatbot import add_message, send
from chatbot.config import ChatConfig

@pytest.fixture
def empty_history() -> list:
    return []

@pytest.fixture
def history() -> list:
    return [{"role": "user", "content": "Hi there!"}]

@pytest.fixture
def system_message() -> list:
    return [{"role": "system", "content": "You are a professional assistant"}]

@pytest.fixture
def mock_response() -> dict:
    return {"message": {"content": "How can I help you?"}}

@pytest.fixture
def chat_config():
    def config(model: str):
        return ChatConfig(model=model)
    return config

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

def test_send_returns_model_reply(chat_config, history, mock_response):

    with patch("chatbot.chatbot.ollama.Client") as MockClient:
        instance = MockClient.return_value
        instance.chat.return_value = mock_response

        result = send(history, chat_config("llama3"))

        assert result == mock_response["message"]["content"]

@patch("chatbot.chatbot.ollama.Client")
def test_send_calls_ollama_with_correct_model(MockClient, chat_config, history, mock_response):
    config = chat_config("mistral")
    MockClient.return_value.chat.return_value = mock_response

    send(history, config)

    MockClient.assert_called_once_with(config.host)
    MockClient.return_value.chat.assert_called_once_with(
        model="mistral",
        messages=history
    )