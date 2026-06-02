"""
hello_claude.py — First API call to Anthropic Claude.

Demonstrates: dotenv setup, system prompt, token usage tracking.
"""

from dotenv import load_dotenv
import anthropic
from anthropic.types import MessageParam

load_dotenv()
client = anthropic.Anthropic()

'''Session A: Hello Claude.'''
msgs_ses_a: list[MessageParam] = [
    {"role": "user", "content": "Say hello in one sentence."}
]

# message = client.messages.create(
#     model="claude-haiku-4-5",
#     max_tokens=100,
#     messages= msgs_ses_a
# )
#
# print(message.content[0].text)

'''Session B: System prompt and token tracking.'''
msgs_ses_b: list[MessageParam] = [
    {"role": "user", "content": "What is a variable?"}
]
system_msg: str = "You are a grumpy teacher who is tired of explaining basic things. Keep answers under 2 sentences."

message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=150,
    system=system_msg,
    messages= msgs_ses_b
)

# Print message content as plain text
print(message.content[0].text)

# Print input and output token counts with the stop reason
print(f"\nTokens:\n- Input: {message.usage.input_tokens}"
      f"\n- Output: {message.usage.output_tokens}"
      f"\n- Stop Reason: {message.stop_reason}")