import logging

import ollama
import pytest

from unittest.mock import patch

import chatbot.chatbot
from chatbot.chatbot import add_message, send, run
from chatbot.config import ChatConfig, setup_logging

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

@patch("chatbot.chatbot.ollama.Client")
def test_send_handles_response_error(MockClient, chat_config, history):
    instance = MockClient.return_value
    instance.chat.side_effect = ollama.ResponseError("Model not found.")

    result = send(history, chat_config("llama3"))

    assert result == "Error: Something went wrong with the model."

@patch("chatbot.chatbot.ollama.Client")
def test_send_handles_unexpected_exception(MockClient, chat_config, history):
    instance = MockClient.return_value
    instance.chat.side_effect = RuntimeError("Connection problem occurred.")

    result = send(history, chat_config("llama3"))

    assert result == "Error: Something went wrong. Is Ollama still running?"

def test_setup_logging(chat_config):
    setup_logging(chat_config("llama3"))

    assert logging.getLogger().level == logging.DEBUG

def test_handlers(chat_config, tmp_path):
    test_log_file = tmp_path / "test_logs.log"
    config = chat_config("llama3")
    config.log_to_file = True

    config.log_file = str(test_log_file)

    setup_logging(config)
    handlers_list = logging.getLogger().handlers
    assert len(handlers_list) == 2
    assert any(isinstance(h, logging.FileHandler) for h in handlers_list)

def test_empty_user_message(chat_config):
    with patch("builtins.input", side_effect=["", "quit"]):
        with patch("builtins.print") as mock_print:
                run(chat_config("llama3"))
                mock_print.assert_any_call(chatbot.chatbot.ENTER_MESSAGE_NOTIFICATION)

@pytest.mark.parametrize("exit_phrase", list(chatbot.chatbot.EXIT_PHRASES))
def test_exit_phrases(chat_config, exit_phrase):
    with patch("builtins.input", side_effect=[exit_phrase]):
        with patch("builtins.print") as mock_print:
            run(chat_config("llama3"))
            mock_print.assert_called_once_with(chatbot.chatbot.AI_GOODBYE_MESSAGE)

def test_send_message(chat_config, mock_response):
    reply = mock_response["message"]["content"]
    with patch("builtins.input", side_effect=["Hello", "quit"]):
        with patch("builtins.print") as mock_print:
            with patch("chatbot.chatbot.send", return_value=reply):
                run(chat_config("llama3"))
                mock_print.assert_any_call(f"{chatbot.chatbot.ASSISTANT.capitalize()}: {reply}")

def test_max_turns(chat_config, mock_response):
    config = chat_config("llama3")
    config.max_turns = 1
    with patch("builtins.input", side_effect=["Hello", "Hi"]):
        with patch("builtins.print") as mock_print:
            with patch("chatbot.chatbot.send", return_value=mock_response["message"]["content"]):
                run(config)
                mock_print.assert_any_call(f"Session reached to limit of {config.max_turns} turns.")
