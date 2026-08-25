from llm_compare.providers.ollama_provider import OllamaProvider


SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided FAQ context. "
    "If the faq_context does not contain enough information to answer, "
    "reply only as 'I don't know' instead of guessing or using outside knowledge. "
    "When you use a source, refer to it by its [FAQ N: <topic>] label."
)


def build_support_prompt(question: str, faq_context: str) -> tuple[str, str]:
    """Assemble a (system_prompt, user_prompt) pair for a FAQ support bot.

    Args:
        question: The customer's question.
        faq_context: Pre-formatted FAQ context (already built elsewhere).

    Returns:
        A (system_prompt, user_prompt) tuple ready to pass into
        provider.ask(user_input=user_prompt, system_prompt=system_prompt).
    """
    system_prompt = SYSTEM_PROMPT
    user_prompt = f"{faq_context}\n\nQuestion: {question}"
    return system_prompt, user_prompt


if __name__ == "__main__":
    faq_context = (
        "[FAQ 1: Shipping Times]\n"
        "Standard shipping takes 3-5 business days within Norway. "
        "Express shipping (additional cost) takes 1-2 business days.\n\n"
        "[FAQ 2: Return Policy]\n"
        "Items can be returned within 30 days of delivery for a full refund, "
        "provided they are unused and in original packaging.\n\n"
        "[FAQ 3: Payment Methods]\n"
        "We accept Visa, Mastercard, and Vipps. Payment is charged at the "
        "time of order confirmation, not at shipment."
    )
    questions = [
        "How long does standard shipping take?",
        "Do you offer a student discount?"
    ]

    provider = OllamaProvider()
    for question in questions:
        system_prompt, user_prompt = build_support_prompt(question, faq_context)
        result = provider.ask(user_input=user_prompt, system_prompt=system_prompt)
        print(f"Question: {question}")
        print(f"Answer  : {result.text}")
        print()
