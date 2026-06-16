import os

import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

message: list[MessageParam] = [
{"role": "user", "content": "Say hello in one sentence"}
]

response = client.messages.create(
    model = os.environ["ANTHROPIC_MODEL_NAME"],
    max_tokens = 100,
    temperature = 0.0,
    messages = message
)

print(f"Reply: {response.content[0].text}")
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")