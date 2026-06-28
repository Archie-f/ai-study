import pytest
from unittest.mock import MagicMock, patch
from provider import LLMResult, ProviderError
from robust_client import with_retry, ProviderChain


def make_result(provider_name: str = "test") -> LLMResult:
    return LLMResult(
        provider=provider_name,
        model="test_model",
        text="hello",
        tokens_in=10,
        tokens_out=5,
        latency_ms=100,
    )


class TestWithRetry:

    def test_retries_on_retryable_error_and_succeeds(self):
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0)  # base_delay=0 → no sleep in tests
        def flaky(prompt: str) -> LLMResult:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderError("test", ValueError("busy"), retryable=True)
            return make_result()

        result = flaky("hi")
        assert call_count == 3
        assert result.text == "hello"

    def test_non_retryable_fails_immediately(self):
        call_count = 0

        @with_retry(max_attempts=3, base_delay=0)
        def always_broken(prompt: str) -> LLMResult:
            nonlocal call_count
            call_count += 1
            raise ProviderError("test", KeyError("content"), retryable=False)

        with pytest.raises(ProviderError):
            always_broken("hi")
        assert call_count == 1  # called exactly once — no retries

    def test_all_retries_exhausted_raises(self):
        @with_retry(max_attempts=3, base_delay=0)
        def always_fails(prompt: str) -> LLMResult:
            raise ProviderError("test", ValueError("busy"), retryable=True)

        with pytest.raises(ProviderError):
            always_fails("hi")


class TestProviderChain:

    def test_fallback_to_second_provider(self):
        primary = MagicMock()
        primary.ask.side_effect = ProviderError(
            "primary", ConnectionError("down"), retryable=False
        )

        secondary = MagicMock()
        secondary.ask.return_value = make_result("secondary")

        chain = ProviderChain([primary, secondary])
        result = chain.ask("hi")

        assert result.text == "hello"
        primary.ask.assert_called_once_with("hi")
        secondary.ask.assert_called_once_with("hi")

    def test_all_providers_fail_raises_provider_error(self):
        primary = MagicMock()
        primary.ask.side_effect = ProviderError("primary", ValueError("busy"), retryable=False)

        secondary = MagicMock()
        secondary.ask.side_effect = ProviderError("secondary", KeyError("content"), retryable=False)

        chain = ProviderChain([primary, secondary])
        with pytest.raises(ProviderError):
            chain.ask("hi")

        primary.ask.assert_called_once_with("hi")
        secondary.ask.assert_called_once_with("hi")