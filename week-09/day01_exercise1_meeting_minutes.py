from docx import Document

def get_agenda_titles(path: str) -> list[str]:
    """
    Checks agenda and returns list of titles

    Args:
        path: path to agenda
    Returns:
         The list of titles
    """
    titles: list[str] = []

    doc = Document(path)
    for paragraph in doc.paragraphs:
        if paragraph.style.name == "Heading 2" and paragraph.text.strip():
            titles.append(paragraph.text.strip())

    return titles