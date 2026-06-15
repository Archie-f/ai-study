import textwrap

billing: str = "billing"
technical: str = "technical"
general: str = "general"
context: str = f"Classify each support ticket as: {billing} / {technical} / {general}"
ticket: str = "Ticket:"
example_list: list[tuple[str, str]] = [
    ("My invoice shows the wrong amount", billing),
    ("The app crashes when I upload a file", technical),
    ("What are your office hours?", general),
]

def build_few_shot_prompt(examples: list[tuple[str, str]], new_input: str) -> str:
    examples_str = "\n".join(f'{ticket} "{inp}" -> {label}' for inp, label in examples)
    return textwrap.dedent(f"""
{context}
    
{examples_str}
    
{ticket} "{new_input}" ->
""")

if __name__ == "__main__":
    input_new: str = "I was charged twice this month"
    print(build_few_shot_prompt(example_list, input_new))