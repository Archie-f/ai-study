def build_review_prompt(customer_message: str, agent_reply: str, policy_summary: str) -> str:
    """Assembles customer message, agent reply, and policy summary into one prompt asking a judge
    to score the reply from 0-3 against the policy.

        Args:
            customer_message: customer message
            agent_reply: agent reply
            policy_summary: policy summary

        Returns:
            Prompt string assembled from the customer message, agent reply, and policy summary
    """
    customer_message = customer_message.strip()
    agent_reply = agent_reply.strip()
    policy_summary = policy_summary.strip()

    return f"""You are an expert quality assurance judge. Your task is to evaluate an agent's compliance with company policy based on a customer interaction.

[POLICY SUMMARY]
{policy_summary}

[CUSTOMER MESSAGE]
{customer_message}

[AGENT REPLY]
{agent_reply}

[EVALUATION CRITERIA]
Score the agent's reply from 0 to 3 using these strict guidelines:
- 3 (Fully Compliant): The agent followed the policy perfectly without any omissions or errors.
- 2 (Mostly Compliant): The agent addressed the core policy requirements but missed minor details.
- 1 (Partially Compliant): The agent attempted to follow policy but made significant errors or omissions.
- 0 (Non-Compliant): The agent completely ignored, violated, or contradicted the policy.

[OUTPUT FORMAT]
Provide your evaluation in JSON format with two keys:
1. "reasoning": A brief explanation justifying the score.
2. "score": An integer from 0 to 3.

JSON Output:"""
