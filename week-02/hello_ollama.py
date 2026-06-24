import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=f"{os.getenv("OLLAMA_BASE_URL")}/v1/",
    api_key=os.getenv("OLLAMA_API_KEY"),
)

response = client.chat.completions.create(
    model=os.getenv("OLLAMA_MODEL_NAME"),
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ],
    max_tokens=150
)

print(response.choices[0].message.content)
print(f"Total used tokens: {response.usage.total_tokens}")
