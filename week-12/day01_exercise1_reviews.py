from dataclasses import dataclass


@dataclass
class ReviewChunk:
    text: str
    product_name: str
    rating: int


def preview_reviews(chunks: list[ReviewChunk]) -> None:
    """Print a one-line preview of each review chunk.

    Args:
        chunks: Review chunks to preview, in the order given.
    """
    for chunk in chunks:
        print(f"Product name: {chunk.product_name:<25} Rating: {chunk.rating:<5} Text: {chunk.text[:40]}..")


if __name__ == "__main__":
    review_chunks = [
        ReviewChunk(
            text="The noise-canceling feature is amazing, but the ear cups get warm after an hour.",
            product_name="AeroSound Pro Headphones",
            rating=4
        ),
        ReviewChunk(
            text="Battery died completely after only two months of light use. Avoid this model.",
            product_name="VoltCharge Power Bank",
            rating=1
        ),
        ReviewChunk(
            text="Sleek design and blazing fast speeds. Best laptop I have ever owned.",
            product_name="ApexBook 14",
            rating=5
        ),
        ReviewChunk(
            text="The picture quality is excellent, though the built-in speakers sound a bit tinny.",
            product_name="Lumix 55-Inch 4K TV",
            rating=4
        ),
        ReviewChunk(
            text="Completely stopped tracking my steps on day three. Returning it immediately.",
            product_name="FitTrack Smart Band",
            rating=1
        ),
        ReviewChunk(
            text="Decent coffee maker for the price. It brews quickly, but the water tank is hard to refill.",
            product_name="BrewMaster Express",
            rating=3
        )
    ]

    preview_reviews(review_chunks)
