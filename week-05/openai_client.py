from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam, ChatCompletion
)

SYSTEM_PROMPT: str = "You are a helpful assistant."
USER_INPUT: str = "Explain what an API is in one sentence."

load_dotenv()
client = OpenAI()

def build_prompt(prompt: str) -> list[ChatCompletionMessageParam]:
    system_turn: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": SYSTEM_PROMPT
    }

    user_turn: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": prompt
    }

    return [system_turn, user_turn]

def call_openai(prompt: str, model: str = 'gpt-4o-mini') -> ChatCompletion:
    """Call OpenAI API and print response text, token counts, and cost.

        Args:
            prompt (str): The prompt to be sent to model.
            model (str, optional): The model to use. Defaults to 'gpt-4o-mini'.
    """

    return client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=build_prompt(prompt),
    )

def print_result(response: ChatCompletion) -> None:
    """Print response text and model prediction score.
    Data used to calculate cost is:
        Pricing for gpt-4o-mini (June 2026):
        Input:  $0.15 per million tokens
        Output: $0.60 per million tokens

        Args:
            response (ChatCompletion): The response to print.
    """
    assert response.choices[0].message.content is not None
    assert response.usage is not None
    text: str = response.choices[0].message.content
    tokens_in: int = response.usage.prompt_tokens
    tokens_out: int = response.usage.completion_tokens

    cost: float = (tokens_in * 0.15 + tokens_out * 0.60) / 1000000

    print(f"""
Response: {text}
Tokens in: {tokens_in} | Tokens out: {tokens_out} | Cost: ${cost:.6f}""")


if __name__ == "__main__":
    print_result(call_openai(USER_INPUT))