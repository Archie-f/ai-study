from dataclasses import dataclass


@dataclass
class BookReference:
    catalog_number: int
    title: str
    author: str


def format_book_references(refs: list[BookReference]) -> str:
    """Format book references as one line per reference.

    Args:
        refs: References to format, in the order given.

    Returns:
        One "[<catalog_number>] <title> by <author>" line per
        reference, joined by newlines, nothing trailing.
    """
    references = [f"[{ref.catalog_number}] {ref.title} by {ref.author}"
                  for ref in refs]
    return "\n".join(references)


if __name__ == "__main__":
    book_references = [
        # 1. Standard entry
        BookReference(
            catalog_number=1001,
            title="To Kill a Mockingbird",
            author="Harper Lee",
        ),
        # 2. Sequential ID, classic novel
        BookReference(catalog_number=1002,
                      title="1984",
                      author="George Orwell"),
        # 3. Multiple authors case
        BookReference(
            catalog_number=1003,
            title="Good Omens",
            author="Neil Gaiman & Terry Pratchett",
        ),
        # 4. Long title with subtitle
        BookReference(
            catalog_number=1004,
            title="The Hobbit, or There and Back Again",
            author="J.R.R. Tolkien",
        ),
        # 5. Non-English name/characters
        BookReference(
            catalog_number=1005,
            title="The Count of Monte Cristo",
            author="Alexandre Dumas",
        ),
        # 6. Sci-Fi/Modern classic
        BookReference(
            catalog_number=1006,
            title="Dune",
            author="Frank Herbert",
        ),
        # 7. Short title, female author pioneer
        BookReference(
            catalog_number=1007,
            title="Frankenstein",
            author="Mary Shelley",
        ),
        # 8. Single-word title, complex narrative
        BookReference(
            catalog_number=1008,
            title="Beloved",
            author="Toni Morrison",
        ),
        # 9. Large catalog number gap (testing sorting/indexing bounds)
        BookReference(
            catalog_number=9999,
            title="The Hitchhiker's Guide to the Galaxy",
            author="Douglas Adams",
        ),
        # 10. Low catalog number/Special format case
        BookReference(
            catalog_number=7,
            title="Pride and Prejudice",
            author="Jane Austen",
        ),
    ]

    print("---")
    print(format_book_references(book_references))
    print("---")
