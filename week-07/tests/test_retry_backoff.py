import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'week-05'))
from provider import ProviderError

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'week-07'))
from retry_backoff import retry_with_backoff

def test_retry_with_backoff_pass_without_retry() -> None:
    def positive():
        return 'success'

    result = retry_with_backoff(positive, max_retries=3, base_delay=0.01)
    assert result == 'success'

def test_retry_with_backoff_fail_with_retry() -> None:
    def negative():
        raise ProviderError(provider_name="test", original_error=Exception("test failure"), retryable=True)

    with pytest.raises(ProviderError, match='test failure') as e:
        retry_with_backoff(negative, max_retries=3, base_delay=0.01)

    assert str(e.value.original_error) == 'test failure'

def test_retry_with_backoff_pass_with_retry() -> None:
    calls: dict[str, int] = {"count": 0}
    def flaky():
        calls['count'] += 1
        if calls['count'] < 3:
            raise ProviderError(provider_name="test", original_error=Exception("rate limited"), retryable=True)
        return 'success'

    result = retry_with_backoff(flaky, max_retries=5, base_delay=0.01)
    assert result == 'success'
    assert calls['count'] == 3