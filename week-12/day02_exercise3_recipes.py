from dataclasses import dataclass


@dataclass
class IngredientReference:
    name: str
    ingredient_index: int
    unit: str
    amount: float

@dataclass
class RecipeSuggestion:
    dish_name: str
    reason: str
    ingredients: list[IngredientReference]
