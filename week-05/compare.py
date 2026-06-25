from datetime import datetime
from dataclasses import dataclass, field
from provider import LLMProvider, LLMResult

TRUNCATE_AT: int = 60

@dataclass
class ComparisonResult:
    """Comparison of the results returned by each LLM provider."""
    prompt: str
    results: list[LLMResult] = field(default_factory=list)
    time_stamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def best_cost(self) -> LLMResult:
        """Get the result that costs least."""
        return min(self.results, key=lambda x: x.cost_usd())

    def fastest(self) -> LLMResult:
        """Get the result that returns fastest."""
        return min(self.results, key=lambda x: x.latency_ms)

    def winner(self) -> LLMResult:
        """Get the result that's both the cheapest and fastest."""
        max_cost = max(r.cost_usd() for r in self.results) or 1.0
        max_latency = max(r.latency_ms for r in self.results) or 1.0

        return min(self.results, key=lambda r: (r.cost_usd() / max_cost) + (r.latency_ms / max_latency))

def run_comparison(
        prompt: str,
        providers: list[LLMProvider],
        system_prompt: str = '',
) -> ComparisonResult:
    """Run prompt across all providers and return a ComparisonResult."""
    comparison_results = ComparisonResult(prompt=prompt)
    for provider in providers:
        result = provider.ask(prompt, system_prompt)
        comparison_results.results.append(result)

    return comparison_results

def truncate(text: str, length: int) -> str:
    """Truncate a string to the given length."""
    return f"{text[:length]}.." if len(text) > length else text

def print_table(comparison: ComparisonResult) -> None:
    """Print a formatted comparison table to stdout."""
    header: str = f"{'Provider':<20} | {'Response':<65} | {'In':<5} | {'Out':<5} | {'Cost ($)':<10} | {'Latency (ms)':<10}"
    divider: str = '-' * len(header)
    winner: LLMResult = comparison.winner()

    print(f"Prompt: {comparison.prompt}")
    print(divider)
    print(header)
    print(divider)

    for result in comparison.results:
        print(f"{result.provider:<20} | {truncate(result.text, TRUNCATE_AT):<65} | {result.tokens_in:<5} | "
              f"{result.tokens_out:<5} | {result.cost_usd():<10.6f} | {result.latency_ms:<10}")
    print(divider)

    fastest: LLMResult = comparison.fastest()
    cheapest: LLMResult = comparison.best_cost()
    print(f"⚡ Fastest: {fastest.provider} ({fastest.latency_ms} ms), 💰 Cheapest: {cheapest.provider} (${cheapest.cost_usd():.6f})")
    print(divider)
    print(f"WINNER: {winner.provider} with Latency of {winner.latency_ms} ms and Cost of ${winner.cost_usd():.6f}")
    print(divider)

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    from anthropic_provider import AnthropicProvider
    from openai_provider import OpenAIProvider
    from ollama_provider import OllamaProvider

    inp: str = "Explain what an API is in one sentence."
    system_message: str = 'You are a helpful assistant.'
    providers_list: list[LLMProvider] = [
        AnthropicProvider(),
        OpenAIProvider(),
        OllamaProvider()
    ]

    results: ComparisonResult = run_comparison(inp, providers_list, system_message)
    print_table(results)