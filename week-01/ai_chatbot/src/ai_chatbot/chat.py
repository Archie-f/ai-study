import asyncio
from dataclasses import dataclass

import httpx

url: str = "http://localhost:11434/api/chat"
model: str = "llama3"

AI_READY_MESSAGE: str = "Chatbot ready. Type 'quit' or 'bye' to exit."
AI_GOODBYE_MESSAGE: str = "AI: Goodbye!"
SYSTEM_PROMPT: str = """
You are a helpful Python mentor.
Explain programming concepts clearly for beginners.
Use simple language and short examples.
""".strip()

type MessageParam = list[dict[str,str]]

system_message: MessageParam = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]
message_history: MessageParam = []

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatConfig:
    url: str = "http://localhost:11434/api/chat"
    model: str = "llama3"
    system_prompt: str = SYSTEM_PROMPT
    max_history: int = 8

def get_user_input() -> str:
    return input("User: ")

def format_message(input_message: str, is_user_input: bool) -> MessageParam:
    role = "assistant"
    if is_user_input:
        role = "user"

    return [{
        "role": role,
        "content": input_message
    }]

def append_message(
        history: MessageParam,
        user_input: str,
        is_user_input: bool
) -> None:
    formatted_message = format_message(user_input, is_user_input)
    history.extend(formatted_message)

def limit_message_history(history: MessageParam) -> MessageParam:
    return history[-8:]

async def send_request_to_model(prompt: MessageParam) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json = {
                    "model": model,
                    "messages": prompt,
                    "stream": False
                },
                timeout=60.0
            )
            # Catch HTTP errors like 404, 500
            response.raise_for_status()
            # Get content from the response. Raise a friendly message if there's none.
            return response.json().get("message", {}).get("content", "Sorry, I didn't get a response. Please try again.")
    except httpx.ConnectError:
        return "Error: Could not connect to Ollama. Is it running?"
    except httpx.HTTPStatusError as err:
        return f"Error: Model returned error: '{err.response.text}'"

def chat() -> None:
    print(AI_READY_MESSAGE)
    while True:
        user_input = get_user_input()

        if user_input.lower() in ["quit", "bye"]:
            print(f"{AI_GOODBYE_MESSAGE}")
            append_message(message_history, user_input, is_user_input=True)
            append_message(message_history, AI_GOODBYE_MESSAGE, is_user_input=False)

            # This is a debug print to see if limiting history works properly
            print(system_message + message_history)
            break

        append_message(message_history, user_input, is_user_input=True)
        prompt = system_message + limit_message_history(message_history)
        ai_reply = asyncio.run(send_request_to_model(prompt))

        print(f"AI: {ai_reply}")
        append_message(message_history, ai_reply, is_user_input=False)

def build_messages(config: ChatConfig, history: list[ChatMessage]) -> MessageParam:
    system_msg: dict[str, str] = {"role": "system", "content": config.system_prompt}
    limited_history: list[ChatMessage] = history[-config.max_history:]
    messages = [{"role": msg.role, "content": msg.content} for msg in limited_history]
    return [system_msg] + messages

if __name__ == "__main__":
    #Test ChatMessage __repr__
    print("########################")
    test_chat_message = ChatMessage("System", SYSTEM_PROMPT)
    print(test_chat_message)
    print("########################")

    #Test build_messages()
    message1: ChatMessage = ChatMessage("user", "What is Python?")
    message2: ChatMessage = ChatMessage("assistant", "Python is a high-level, interpreted programming language.")
    message3: ChatMessage = ChatMessage("user", "Is it good for beginners?")
    message4: ChatMessage = ChatMessage("assistant", "Yes, its simple syntax makes it excellent for beginners.")
    message5: ChatMessage = ChatMessage("user", "Can you show me a hello world example?")
    message6: ChatMessage = ChatMessage("assistant", "Yes, print('Hello world!')")

    test_history = [message1, message2, message3, message4, message5, message6]
    message_built: MessageParam = build_messages(ChatConfig(), test_history)
    print(message_built)
    print("########################")

    #Start chat
    chat()