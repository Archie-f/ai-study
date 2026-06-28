import time

import anthropic
from anthropic.types import MessageParam

from provider import *

class AnthropicProvider(LLMProvider):
    """Anthropic Claude via the Anthropic SDK."""

    def __init__(self, model: str = 'claude-haiku-4-5') -> None:
        self.client = anthropic.Anthropic()
        self.model = model

    def ask(self, user_input: str, system_prompt: str = '') -> LLMResult:
        """Call Anthropic messages create and return unified LLMResult.

        Args:
            user_input: The text prompt provided by the user.
            system_prompt: Optional background instructions for the model.

        Returns:
            A structured LLMResult object containing unified metrics.
"""
        prompt: list[MessageParam] = [
            {"role": "user", "content": user_input}
        ]
        start_time: float = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=system_prompt,
                messages=prompt,
            )
            elapsed_time = (time.perf_counter() - start_time) * 1000
            return LLMResult(
                provider='claude',
                model=self.model,
                text=response.content[0].text,
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
                latency_ms=round(elapsed_time),
            )
        except anthropic.RateLimitError as e:
            raise ProviderError(
                provider_name="anthropic",
                original_error=e,
                retryable=True,
            ) from e
        except anthropic.APITimeoutError as e:
            raise ProviderError(
                provider_name="anthropic",
                original_error=e,
                retryable=True,
            ) from e
        except (KeyError, AttributeError) as e:
            raise ProviderError(
                provider_name="anthropic",
                original_error=e,
                retryable=False,
            ) from e
