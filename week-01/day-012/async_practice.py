import asyncio
import time

import httpx


# Exercise 4
# Create async_practice.py and copy the slow_task / main example above.
# Run it with just the sequential version. Time it with: time python3 async_practice.py
# Switch to the gather() version. Time it again. Observe the difference.
# Add a fourth task: slow_task('D', 1.5). Does the total time change?
# Add print statements inside slow_task to see the order tasks start and finish.
#   — Do they start in the order you listed them in gather()?
#   — Do they finish in the same order?

async def slow_task(name: str, seconds: float) -> str:
    print(f"Task {name}: starting")
    await asyncio.sleep(seconds)
    print(f"Task {name}: done")
    return f"Task {name} finished after {seconds} seconds."

async def main():
    start_time = time.time()

    results = await asyncio.gather(
        slow_task("A", 1),
        slow_task("B", 2),
        slow_task("X", 2.3),
        slow_task("C", 3),
        slow_task("D", 1.5)
    )
    print(*results, sep="\n")
    print(f"Total time: {(time.time() - start_time):.2f} seconds.")

asyncio.run(main())
modelA = 'llama3'
modelB = 'mistral'
message = 'Explain a Python list comprehension in one sentence.'

async def ask_model(model: str, prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json = {
                "model": model,
                "messages": [{'role': 'user', 'content': prompt}],
                "stream": False
            },
            timeout=60.0
        )
        return response.json().get("message", {}).get("content", "")

async def compare_models(model1: str, model2: str, prompt: str) -> None:
    responses =  await asyncio.gather(
        ask_model(model1, prompt),
        ask_model(model2, prompt)
    )
    print(f'\n--- {model1} ---')
    print(responses[0])
    print(f'\n--- {model2} ---')
    print(responses[1])

if __name__ == '__main__':
    asyncio.run(
        compare_models(modelA, modelB, message)
    )
