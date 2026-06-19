import textwrap

def build_prompt(task: str, context: str, output_format: str) -> str:
    """Build prompt text with the given parameters.
        Args:
            task (str): Task
            context (str): Context
            output_format (str): Output format

        Returns:
            str: Prompt text
    """
    return textwrap.dedent(f"""
    Context: {context}

    Task: {task}

    Output format: {output_format}
    """).strip()

if __name__ == "__main__":
    context: str = "The codebase uses Python 3.12+"
    task: str = "Review the function below and share your feedback"
    output_format: str = "HTML"

    print(build_prompt(task=task, context=context, output_format=output_format))