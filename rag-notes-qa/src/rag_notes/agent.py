from dataclasses import dataclass, field

from rag_notes.generate import build_context
from rag_notes.protocols import Retriever


@dataclass
class AgentResult:
    """What run_agent returns."""
    answer: str
    steps: int        # how many times the model is called
    tokens_in: int
    tokens_out: int
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    stop_reason: str = "end_turn"


class AgentStepLimitError(Exception):
    """Raised when the model still wants tools after max_steps model calls."""


SEARCH_NOTES_TOOL = {
    "name": "search_notes",
    "description": (
        "Search the user's daily takeaway notes. It covers Python basics, multi-provider LLM script, "
        "GitHub README and CI, prompt engineering, LLM API integration, structured outputs, system prompts, "
        "git (commits, resetting, branches), multi-turn conversations, OpenAI API, provider abstraction, "
        "Ollama and local models, evaluation and scoring, eval datasets, LLM judge, failure analysis, "
        "token economics and pricing, streaming, cost and latency dashboard, model selection, resilience, "
        "HTML reports, package refactor, README and documentation, publishing, "
        "chunking, golden Q&A dataset, embeddings, Chroma vector store, BM25, hybrid search, "
        "LangChain and LlamaIndex reading, citations, end-to-end CLI, guardrails, retrieval metrics (Recall@k, MRR), "
        "LLM-as-judge faithfulness, eval harness, Protocol-based interfaces. "
        "Use it for any question whose answer could be in the user's daily takeaway notes, even if the question "
        "does not name the topic. Do not call it for questions unrelated to the user's daily takeaway notes. "
        "Return the passages that are the most relevant to the question, or 'No results' if nothing matches."
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
    """Collect the tool requests from one model response.

    Args:
        content: response.content, a list of blocks (text, tool_use, ...).

    Returns:
        (tool_use_id, tool_name, tool_input) for every tool_use block,
        in the order the model sent them. Blocks of any other type are
        ignored; no tool_use blocks -> [].
    """
    return [(block.id, block.name, block.input) for block in content if block.type == "tool_use"]


def make_tool_result_message(results: list[tuple[str, str, bool]]) -> dict:
    """Build the single user message that carries all tool results back to the model.

    Args:
        results: (tool_use_id, content, is_error) per tool call, in the
            same order as the requests.

    Returns:
        {"role": "user", "content": [...]} with one tool_result block per
        result, order preserved. "is_error" is set only on failed results.

    Raises:
        ValueError: if results is empty. A user message with empty content
            is invalid and would leave the model waiting for its tool results.
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
    """Run one tool call and turn every outcome into a (content, is_error) pair.

    The model can send any name and any input, so nothing here is trusted
    and nothing is raised: failures go back to the model as error results
    and the loop continues.

    Args:
        name: Tool name requested by the model. Only "search_notes" exists.
        tool_input: Input dict from the model. "query" (non-blank str) is
            required; "n" (int, 1..10, bool not accepted) defaults to 5.
        retriever: Used for the search.

    Returns:
        (content, is_error):
        - unknown name, bad query, bad n -> (message, True)
        - search finds nothing -> ("No results for query: ...", False),
          an empty search is an answer, not an error
        - results found -> (build_context(results), False)
        - the retriever raises any exception -> ("Error running search: ...", True)
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
        tool: dict | None = None
) -> AgentResult:
    """Answer a question by letting the model call search_notes until it replies with text.

    Each loop iteration is one model call. If the model asks for tools, they
    are run, their results are sent back in one user message and the loop
    continues; any other stop_reason ends the loop with the text answer.

    Args:
        question: The user's question.
        client: Anything with messages.create(...) in the Anthropic format.
        retriever: Used by the search_notes tool.
        llm_model: Name of the Claude model (not the embedding model).
        max_steps: Maximum number of model calls.
        max_tokens: Output token limit for each model call.
        tool: Tool definition sent to the model. None -> SEARCH_NOTES_TOOL.

    Returns:
        AgentResult with the answer text, the number of model calls, token
        totals summed over all calls, every tool request the model made and the stop_reason: why
        the last model call ended; anything other than ‘end_turn’ means the answer may be incomplete or refused.

    Raises:
        AgentStepLimitError: if the model still asks for tools after
            max_steps calls.
    """
    messages = [{"role": "user", "content": question}]
    tokens_in = 0
    tokens_out = 0
    tool_calls = []

    if tool is None:
        tool = SEARCH_NOTES_TOOL

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=llm_model,
            max_tokens=max_tokens,
            tools=[tool],
            messages=messages
        )
        tokens_in += response.usage.input_tokens
        tokens_out += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            answer = "".join(block.text for block in response.content if block.type == "text")
            return AgentResult(
                answer=answer,
                steps=step,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                tool_calls=tool_calls,
                stop_reason=response.stop_reason,
            )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = parse_tool_uses(response.content)

        results = []
        for tool_use_id, tool_name, tool_input in tool_uses:
            content, is_error = run_tool(tool_name, tool_input, retriever)
            results.append((tool_use_id, content, is_error))
            tool_calls.append((tool_name, tool_input))

        messages.append(make_tool_result_message(results))

    raise AgentStepLimitError(f"Model still requested tools after {max_steps} steps")
