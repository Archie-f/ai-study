import asyncio
import time

import httpx

# Exercise 5
# In your chatbot project, create a new file: src/ai_chatbot/async_chat.py
# Write the ask_model(model, prompt) async function from the pattern above.
# Write compare_models(prompt) that calls ask_model for llama3 twice with different prompts.
#   (Use the same model twice if you only have llama3 — asyncio.gather still runs them concurrently.)
# Add asyncio.run(compare_models('What is 2 + 2?')) at the bottom.
# Run it: python3 src/ai_chatbot/async_chat.py
# Bonus: Use time.time() before and after asyncio.gather() to measure how long two calls take vs one sequential call.

URL = "http://localhost:11434/api/chat"
MODEL = "llama3"

promptA = "Explain tuple in Python in one short sentence."
promptB = "What is python? Explain in one short sentence."

async def ask_model(model: str, prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=60.0
        )
        return response.json().get("message", {}).get("content", "")

async def compare_responses(prompt1: str, prompt2: str) -> None:
    start = time.time()
    responses = await asyncio.gather(
        ask_model(MODEL, prompt1),
        ask_model(MODEL, prompt2)
    )
    end = time.time()
    print(*responses, sep="\n")
    print(f"Time taken: {(end - start):.2f} seconds.")

if "__main__" == __name__:
    asyncio.run(compare_responses(promptA, promptB))
    question = "What is 2 + 2?"
    asyncio.run(compare_responses(question, question))