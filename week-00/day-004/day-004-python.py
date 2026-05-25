SYSTEM_PROMPT = """
You are a helpful Python mentor.
Explain programming concepts clearly for beginners.
Use simple language and short examples.
""".strip()

def format_history(history):
    return '\n'.join(history)

def get_user_input():
    return input('User: ')

def build_prompt(history, user_input):
    formatted_history = format_history(history)
    return f"""
System: {SYSTEM_PROMPT}
{formatted_history}
User: {user_input}
AI:
"""

history = [
    "User: Hello",
    "AI: Hi!"
]

print(build_prompt(history, get_user_input()))