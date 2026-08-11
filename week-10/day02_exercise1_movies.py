from pathlib import Path

import chromadb


def make_movie_collection(persist_path: str):
    """Create (or open) a cosine-configured Chroma collection named
    "movie_plots" at persist_path, and return it.

    Args:
        persist_path: folder to store the Chroma database in
    Returns:
        the collection, ready to have items added
    """
    client = chromadb.PersistentClient(path=persist_path)
    collection = client.get_or_create_collection(
        name="movie_plots",
        configuration={"hnsw": {"space": "cosine"}},
    )

    print(client.list_collections())
    print(collection.count())


if __name__ == "__main__":
    path = str(Path(__file__).parent / "persistent")
    make_movie_collection(path)
