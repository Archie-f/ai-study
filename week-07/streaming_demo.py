import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / 'week-05'))
from provider import LLMResult

def print_stream_and_collect_result(generator) -> LLMResult:
    try:
        while True:
            print(next(generator), end="", flush=True)
    except StopIteration as end:
        print("\n----------------------------------------")
        print(f" * Response: \n{end.value.text}", flush=True)
        print(f" * Tokens In: {end.value.tokens_in}", flush=True)
        print(f" * Tokens Out: {end.value.tokens_out}", flush=True)
        print(f" * Elapsed Time: {end.value.latency_ms}", flush=True)
        return end.value


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from anthropic_provider import AnthropicProvider

    provider = AnthropicProvider()
    gen = provider.ask_stream("Write a two-line limerick about deploying code on a Friday.")
    print_stream_and_collect_result(gen)