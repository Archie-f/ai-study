import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.getenv("GROQ_BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL_NAME"),
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ],
    max_tokens=150
)

print(response.choices[0].message.content)
print(f"Token usage: {response.usage.total_tokens}")