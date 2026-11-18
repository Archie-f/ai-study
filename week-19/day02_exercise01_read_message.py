from langchain_core.messages import AIMessage

def requested_tools(message: AIMessage) -> list[tuple[str, dict]]:
    """Return (name, args) for every tool call in the message, in order.

    Returns [] when the model asked for no tool.
    """
    return [(call["name"], call["args"]) for call in message.tool_calls]