def classify_comment(text: str) -> str:
    ...  # returns "safe" or "flagged"


def build_test_cases() -> list[tuple[str, str]]:
    """Build a categorized set of (input, expected_label) test cases.

    Returns:
        A list covering at least a clearly safe case, a clearly
        flagged case, a borderline case, and an adversarial case
        written to try to evade classification.
    """
    return [
        ("This tutorial really helped me understand recursion, thanks!", "safe"),
        ("You're an idiot and everyone here is worthless.", "flagged"),
        ("This is actually kind of dumb, not gonna lie.", "flagged"),
        ("You're an 1d10t and w0rthless", "flagged"),
    ]