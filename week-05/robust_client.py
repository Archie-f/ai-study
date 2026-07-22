import logging


from provider import LLMProvider, LLMResult, ProviderError

logger = logging.getLogger(__name__)

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
        assert last_error is not None
        raise ProviderError(
            provider_name="ProviderChain",
            original_error=RuntimeError("All providers failed"),
            retryable=False,
        ) from last_error
