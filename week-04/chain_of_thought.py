import textwrap

POLICY: str = "Refunds allowed within 30 days."
COT_INSTRUCTIONS: str = """
Think step by step:
1. What does the policy say?
2. Does the request meet the condition?
3. What is the final answer?
""".strip()


def build_cot_prompt(policy: str, request: str) -> str:
    """Build a cot prompt.
        Args:
            policy (str): Policy text
            request (str): Request text
    """
    return textwrap.dedent(f"""
Policy: {policy}
    
Request: "{request}"
    
{COT_INSTRUCTIONS}
    
Reasoning:
Answer: """)

if __name__ == "__main__":
    request: str = "I bought the wrong size and it's been 45 days."
    print(build_cot_prompt(POLICY, request))