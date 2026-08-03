import re

_PATTERN = re.compile(r"module-(\d+)-lesson-(\d+)-notes\.docx")

def parse_module_lesson(filename: str) -> tuple[int | None, int | None]:
    """Parse a module lesson file.
        Args:
            filename: module lesson file name.
        Returns:
            The numbers of module and lesson in the given file nam in a tuple of strings.
    """
    result = _PATTERN.match(filename)

    return (int(result.group(1)), int(result.group(2))) if result else (None, None)