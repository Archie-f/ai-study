import requests

url = "http://localhost:11434/api/generate"
model = "llama3"
SYSTEM_PROMPT = """
You are a helpful Python mentor.
Explain programming concepts clearly for beginners.
Use simple language and short examples.
""".strip()
history_text = []
goodbye_text = "AI: Goodbye!"

def get_user_input():
    return input("User: ")

def format_history(history):
    return '\n'.join(history[-4:])

def build_prompt(system_prompt, history, user_input):
    return f"""
System: {system_prompt}
{format_history(history)}
User: {user_input}
AI:
"""

def ask_model(prompt):
    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
    return response.json().get("response")

def append_input(history, input_text, is_user_input):
    if is_user_input:
        history.append(f'User: {input_text}')
    else:
        history.append(f'AI: {input_text}')

def day_four_chat_bot():
    while True:
        user_input = get_user_input()
        if user_input == 'quit' or user_input == 'bye':
            append_input(history_text, user_input, is_user_input=True)
            history_text.append(goodbye_text)
            print(f'{goodbye_text} \nHistory: {history_text}')
            break

        prompt = build_prompt(SYSTEM_PROMPT, history_text, user_input)
        answer = ask_model(prompt)

        append_input(history_text, user_input, is_user_input=True)
        append_input(history_text, answer, is_user_input=False)
        print(f'AI: {answer}')

day_four_chat_bot()