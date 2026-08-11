from pathlib import Path

import chromadb
from chromadb import Collection

from rag_notes.embedder import load_embedding_model

PATH = str(Path(__file__).parent / "persistent")
COLLECTION_NAME = "recipe_collection"
def make_recipe_collection(path: str, name: str = COLLECTION_NAME) -> Collection:
    """Make a recipe collection from a path.
    Args:
        path: path to recipe collection
        name: name of recipe collection
    Returns:
        collection of recipes
    """
    client = chromadb.PersistentClient(path)
    return client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}}
    )


def build_recipe_metadata(recipe: dict) -> dict:
    """Convert a raw recipe dict into a Chroma-safe metadata dict.

    Args:
        recipe: dict with keys "title", "cuisine", "prep_time_minutes" (may be None)
    Returns:
        dict with only str | int | float | bool values, no None
    """
    return {
        "title": recipe["title"],
        "cuisine": recipe["cuisine"],
        "prep_time_minutes": recipe["prep_time_minutes"] or -1,
    }


def add_recipes(collection, recipes_list: list[dict]) -> None:
    """Add recipes to the recipes list.
    Args:
        recipes_list: list of recipes
        collection: collection of recipes
    """
    ids = [f"recipe-{index}" for index in range(len(recipes_list))]
    print(ids)

    model = load_embedding_model()
    titles = [recipe["title"] for recipe in recipes_list]
    vectors = model.encode(titles)

    collection.add(
        ids=ids,
        embeddings=vectors,
        metadatas=[build_recipe_metadata(recipe) for recipe in recipes_list],
        documents=[recipe['cuisine'] for recipe in recipes_list],
    )


recipes = [
    {"title": "Lentil Soup", "cuisine": "Middle Eastern", "prep_time_minutes": 15},
    {"title": "Grandma's Secret Stew", "cuisine": "Home Cooking", "prep_time_minutes": None},
    {"title": "Caprese Salad", "cuisine": "Italian", "prep_time_minutes": 10},
]

if __name__ == "__main__":
    collection = make_recipe_collection(PATH)
    add_recipes(collection, recipes)
