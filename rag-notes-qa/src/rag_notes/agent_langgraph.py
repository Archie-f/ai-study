from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import MessagesState, StateGraph, START, END

from rag_notes.agent import SEARCH_NOTES_TOOL, run_tool, AgentResult, AgentStepLimitError
from rag_notes.protocols import Retriever

def route_after_model(state: MessagesState) -> str:
    """Decide where the graph goes after a model call.

    Runs after call_model, so the last message is always the model's
    AIMessage. The decision looks at its tool_calls, the same list
    run_tools will execute.

    Args:
        state: Current graph state; only the last message is read.

    Returns:
        "tools" if the last message asks for at least one tool, otherwise "end".
    """
    return "tools" if state["messages"][-1].tool_calls else "end"


def build_agent_graph(llm, retriever: Retriever, tool: dict | None = None):
    """Build and compile the agent graph around one model and one retriever.

    The tool is bound to the model once. call_model and run_tools are
    closures: LangGraph calls them with the state only, and they reach
    the bound model and the retriever through this function's scope.
    Graph: START -> call_model -> (route_after_model) -> run_tools -> call_model,
    or END when the model asks for no tool.

    Args:
        llm: Chat model with bind_tools(tools) returning an object whose
            invoke(messages) returns an AIMessage (ChatAnthropic, or a fake).
        retriever: Used by run_tool for the search_notes tool.
        tool: Tool definition sent to the model. None -> SEARCH_NOTES_TOOL.

    Returns:
        The compiled graph; call invoke({"messages": [...]}) on it.
    """
    if tool is None:
        tool = SEARCH_NOTES_TOOL

    llm_with_tools = llm.bind_tools([tool])

    def call_model(state: MessagesState) -> dict:
        """Send the whole conversation to the model.

        Args:
            state: Current graph state; all messages are sent.

        Returns:
            {"messages": [answer]}: only the new AIMessage; add_messages
            appends it to the conversation.
        """
        answer = llm_with_tools.invoke(state["messages"])
        return {"messages": [answer]}

    def run_tools(state: MessagesState) -> dict:
        """Run every tool call of the last AIMessage through run_tool.

        Every call gets an answer, in one return: the API rejects the next
        model call if any tool request is left without a result.

        Args:
            state: Current graph state; only the last message is read.

        Returns:
            {"messages": [...]}: one ToolMessage per call, in order, with the
            call's id as tool_call_id and status="error" when run_tool
            reported an error, otherwise "success".
        """
        last_message = state["messages"][-1]

        messages = []
        for call in last_message.tool_calls:
            content, is_error = run_tool(call["name"], call["args"], retriever)
            tool_message = ToolMessage(
                content=content,
                tool_call_id=call["id"],
                status="error" if is_error else "success",
            )
            messages.append(tool_message)

        return {"messages": messages}

    builder = StateGraph(MessagesState)             # type: ignore
    builder.add_node("call_model", call_model)      # type: ignore
    builder.add_node("run_tools", run_tools)        # type: ignore
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {"tools": "run_tools", "end": END}
    )
    builder.add_edge("run_tools", "call_model")
    return builder.compile()


def run_langgraph_agent(
        question: str,
        llm,
        retriever: Retriever,
        max_steps: int = 5,
        tool: dict | None = None,
) -> AgentResult:
    """Answer a question with the LangGraph agent; same result type as run_agent.

    The numbers are read from the final conversation: every AIMessage is
    one model call. recursion_limit counts node runs (call_model and
    run_tools are one round), so 2 * max_steps stops at the same point
    as run_agent: max_steps model calls and max_steps tool runs.

    Args:
        question: The user's question.
        llm: Chat model, see build_agent_graph.
        retriever: Used by the search_notes tool.
        max_steps: Maximum number of model calls.
        tool: Tool definition sent to the model. None -> SEARCH_NOTES_TOOL.

    Returns:
        AgentResult with the text of the last AIMessage, the number of model
        calls, token totals summed over all calls, every (name, args) tool
        request in order and the stop_reason of the last call; anything other
        than "end_turn" means the answer may be incomplete or refused.

    Raises:
        AgentStepLimitError: if the model still asks for tools after
            max_steps model calls (raised from GraphRecursionError).
    """
    graph = build_agent_graph(llm, retriever, tool)
    try:
        final_state = graph.invoke(
            {"messages": [HumanMessage(question)]},     # type: ignore
            config={"recursion_limit": 2 * max_steps},        # type: ignore
        )
    except GraphRecursionError as err:
        raise AgentStepLimitError(f"Model still requested tools after {max_steps} steps") from err

    messages = final_state["messages"]
    ai_messages = [message for message in messages if isinstance(message, AIMessage)]
    tool_calls = []
    for ai_message in ai_messages:
        calls = [(call["name"], call["args"]) for call in ai_message.tool_calls]
        tool_calls.extend(calls)

    return AgentResult(
        answer=ai_messages[-1].text,
        steps=len(ai_messages),
        tokens_in=sum(m.usage_metadata["input_tokens"] for m in ai_messages),
        tokens_out=sum(m.usage_metadata["output_tokens"] for m in ai_messages),
        tool_calls=tool_calls,
        stop_reason=ai_messages[-1].response_metadata["stop_reason"]
    )
