import requests

url = "http://localhost:11434/api/chat"
model = "llama3"

AI_READY_MESSAGE = "Chatbot ready. Type 'quit' or 'bye' to exit."
AI_GOODBYE_MESSAGE = "AI: Goodbye!"
SYSTEM_PROMPT = """
You are a helpful Python mentor.
Explain programming concepts clearly for beginners.
Use simple language and short examples.
""".strip()

system_message = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]
message_history = []

def get_user_input():
    return input("User: ")

def format_message(input_message, is_user_input):
    role = "assistant"
    if is_user_input:
        role = "user"

    return [{
        "role": role,
        "content": input_message
    }]

def append_message(history, user_input, is_user_input):
    formatted_message = format_message(user_input, is_user_input)
    history.extend(formatted_message)

def limit_message_history(history):
    return history[-8:]

def send_request_to_model(prompt):
    try:
        response = requests.post(
            url,
            json={
                "model": model,
                "messages": prompt,
                "stream": False
            }
        )
        # Catch HTTP errors like 404, 500
        response.raise_for_status()
        # Get content from the response. Raise a friendly message if there's none.
        return response.json().get("message", {}).get("content", "Sorry, I didn't get a response. Please try again.")

    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is it running?"
    except requests.exceptions.HTTPError as err:
        return f"Error: Model returned error: '{err.response.text}'"

def chat():
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
        ai_reply = send_request_to_model(prompt)

        print(f"AI: {ai_reply}")
        append_message(message_history, ai_reply, is_user_input=False)

chat()
