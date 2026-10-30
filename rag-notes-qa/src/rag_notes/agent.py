from dataclasses import dataclass

from rag_notes.generate import build_context
from rag_notes.protocols import Retriever


@dataclass
class AgentResult:
    """What run_agent returns."""
    answer: str
    steps: int        # how many times the model is called
    tokens_in: int
    tokens_out: int


class AgentStepLimitError(Exception):
    """Raised when the model still wants tools after max_steps model calls."""


SEARCH_NOTES_TOOL = {
    "name": "search_notes",
    "description": (
        "Search the user's daily takeaway notes and return the passages that are the most relevant to the question. "
        "Use it for any question about what the notes say. "
        "Do not use it for any general knowledge questions that the notes do not cover."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search terms, phrase, or topic to look up within the personal notes."
            },
            "n": {
                "type": "integer",
                "default": 5,
                "description": "The maximum number of most relevant notes to retrieve from the search results.",
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["query"]
    }
}


def parse_tool_uses(content: list) -> list[tuple[str, str, dict]]:
    """Return (tool_use_id, tool_name, tool_input) for every tool_use block, in order.

    Blocks of any other type (text, ...) are ignored. No tool_use blocks -> [].
    """
    return [(block.id, block.name, block.input) for block in content if block.type == "tool_use"]


def make_tool_result_message(results: list[tuple[str, str, bool]]) -> dict:
    """Build the single user message that carries all tool results back to the model.

    results: (tool_use_id, content, is_error) per tool call, in the same order.
    Each becomes one tool_result block. Order is preserved.
    """
    if not results:
        raise ValueError("results must not be empty")

    content = []
    for tool_use_id, content_text, is_error in results:
        each_dict: dict = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content_text
        }

        if is_error:
            each_dict["is_error"] = is_error

        content.append(each_dict)

    return {
        "role": "user",
        "content": content
    }


def run_tool(name: str, tool_input: dict, retriever: Retriever) -> tuple[str, bool]:
    """Run one tool call. Returns (content, is_error).

    - unknown tool name        -> ("Unknown tool: <name>", True)
    - search_notes:
        * query missing, not a str, or blank  -> (message, True)
        * n missing                           -> default 5
        * n not an int (bool does not count) or outside 1..10 -> (message, True)
        * no results from retriever.search    -> ("No results", False)
        * otherwise                           -> (build_context(results), False)
    - any exception raised while running -> (str of the error, True); never propagate
    """
    if name != "search_notes":                              #Unknown tool name -> ("Unknown tool: <name>",True)
        return f"Unknown tool: {name}", True

    query = tool_input.get("query")
    if not isinstance(query, str) or not query.strip():     #If "query" is missing, not a str, or blank -> (message,True)
        return f"query must be a valid string: {query}", True

    n = tool_input.get("n", 5)                  #If "n" is missing -> default 5

    if type(n) is not int or not (1 <= n <= 10):            #"n" not an int (bool does not count) or outside 1..10 -> (message,True)
        return f"n must be a valid integer: {n}", True

    try:
        results = retriever.search(query, n)
        if not results:                                     #If no results from retriever.search -> ("No results", False)
            return f"No results for query: {query}", False

        return build_context(results), False
    except Exception as e:
        return f"Error running search: {e}", True           #Any exception raised -> (error message, True)


def run_agent(
        question: str,
        client,
        retriever: Retriever,
        llm_model: str,
        max_steps: int = 5,
        max_tokens: int = 1024,
) -> AgentResult:
    """Answer question by letting the model call search_notes until it replies with text.

    Each loop iteration is one model call:
    - call client.messages.create(model=, max_tokens=, tools=[SEARCH_NOTES_TOOL], messages=)
    - add response.usage.input_tokens / output_tokens to the totals
    - stop_reason != "tool_use": return AgentResult(answer=<text of the text blocks joined>, steps=<model calls so far>, tokens_in, tokens_out)
    - stop_reason == "tool_use": append the assistant message (response.content unchanged),
      run every tool_use via run_tool, append ONE user message built by make_tool_result_message
    After max_steps model calls that all ended in tool_use: raise AgentStepLimitError.
    """
    messages = [{"role": "user", "content": question}]
    tokens_in = 0
    tokens_out = 0

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=llm_model,
            max_tokens=max_tokens,
            tools=[SEARCH_NOTES_TOOL],
            messages=messages
        )
        tokens_in += response.usage.input_tokens
        tokens_out += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            answer = "".join(block.text for block in response.content if block.type == "text")
            return AgentResult(answer=answer, steps=step, tokens_in=tokens_in, tokens_out=tokens_out)

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = parse_tool_uses(response.content)

        results = []
        for tool_use_id, tool_name, tool_input in tool_uses:
            content, is_error = run_tool(tool_name, tool_input, retriever)
            results.append((tool_use_id, content, is_error))

        messages.append(make_tool_result_message(results))

    raise AgentStepLimitError(f"Model still requested tools after {max_steps} steps")
