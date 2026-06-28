import logging
import time
import functools
from typing import Callable, TypeVar
from provider import LLMProvider, LLMResult, ProviderError

T = TypeVar("T")
logger = logging.getLogger(__name__)


def with_retry(max_attempts: int = 3, base_delay: float = 1.0) -> Callable:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_error: ProviderError | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except ProviderError as e:
                    last_error = e
                    if not e.retryable:
                        raise  # fail immediately, don't retry
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"  Retry {attempt + 1}/{max_attempts - 1} "
                              f"after {delay:.1f}s — {e}")
                        time.sleep(delay)
            raise last_error  # all attempts exhausted
        return wrapper
    return decorator

class ProviderChain:
    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("ProviderChain requires at least one provider")
        self._providers = providers

    def ask(self, prompt: str) -> LLMResult:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                result = provider.ask(prompt)
                logger.info(
                    "Success: %s answered in %dms",
                    type(provider).__name__,
                    result.latency_ms,
                )
                return result
            except ProviderError as e:
                logger.warning(
                    "Provider %s failed (%s), trying next",
                    e.provider_name,
                    type(e.original_error).__name__,
                )
                last_error = e
        raise ProviderError(
            provider_name="ProviderChain",
            original_error=RuntimeError("All providers failed"),
            retryable=False,
        ) from last_error