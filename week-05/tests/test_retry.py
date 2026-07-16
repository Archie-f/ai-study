import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock
from src.llm_compare.providers.base import LLMResult, ProviderError

sys.path.append(str(Path(__file__).resolve().parent.parent / 'week-05'))
from robust_client import ProviderChain


def make_result(provider_name: str = "test") -> LLMResult:
    return LLMResult(
        provider=provider_name,
        model="test_model",
        text="hello",
        tokens_in=10,
        tokens_out=5,
        latency_ms=100,
    )


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