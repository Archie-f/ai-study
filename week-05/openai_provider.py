import time

import openai
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)

from provider import *


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = 'gpt-4o-mini') -> None:
        """Initialize OpenAI client and store model name."""
        self.client = OpenAI()
        self.model = model

    def ask(self, user_input: str, system_prompt: str = '') -> LLMResult:
        """Call OpenAI chat completions and return unified LLMResult.

            Args:
                user_input: Input to ask user to enter chat.
                system_prompt: Input to ask user to enter chat.

            Returns:
                LLMResult: Result of asking user to enter chat.
        """
        system_turn: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_prompt
        }
        user_turn: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": user_input
        }
        prompt: list[ChatCompletionMessageParam] = [system_turn, user_turn]
        start_time: float = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                messages=prompt,
            )
            elapsed_time: float = (time.perf_counter() - start_time) * 1000
            return LLMResult(
                provider='open_ai',
                model=self.model,
                text=response.choices[0].message.content,
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
                latency_ms=round(elapsed_time),
            )
        except openai.RateLimitError as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=True
            ) from e
        except openai.APITimeoutError as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=True
            ) from e
        except (KeyError, AttributeError) as e:
            raise ProviderError(
                provider_name="openai",
                original_error=e,
                retryable=False
            ) from e
