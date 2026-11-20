from langgraph.graph import StateGraph, START, END

from day01_exercise02_cart_nodes import CartState, compute_subtotal, apply_discount

def build_cart_graph():
    """Compile the cart pipeline.
    Route: START -> compute_subtotal -> apply_discount -> END.

    Returns: a compiled graph; call .invoke(initial_state) on it.
    """
    builder = StateGraph(CartState)                     # type: ignore
    builder.add_node("subtotal", compute_subtotal)      # type: ignore
    builder.add_node("apply_discount", apply_discount)  # type: ignore
    builder.add_edge(START, "subtotal")
    builder.add_edge("subtotal", "apply_discount")
    builder.add_edge("apply_discount", END)
    return builder.compile()


if __name__ == "__main__":
    state_normal: CartState = {"items": [(10.0, 2), (5.0, 1)], "subtotal": 0.0, "total": 0.0}
    state_below_20: CartState = {"items": [(4.0, 2)], "subtotal": 0.0, "total": 0.0}
    states = [state_normal, state_below_20]

    graph = build_cart_graph()
    results = []
    for state in states:
        result = graph.invoke(state)                    # type: ignore
        results.append(result)

    print(results)
