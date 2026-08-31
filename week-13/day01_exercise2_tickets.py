def category_reciprocal_rank(predicted_categories: list[str], true_category: str) -> float:
    """Returns the reciprocal rank of a given category.

    Args:
        predicted_categories (list[str]): list of predicted categories
        true_category (str): true category
    Returns:
        1 / rank of true_category in predicted_categories, or 0.0 if it's absent.
    """
    return 1 / (1 + predicted_categories.index(true_category)) if true_category in predicted_categories else 0.0


if __name__ == "__main__":
    predicted_categories = ['billing', 'shipping', 'refund', 'login']
    true_category = 'refund'
    false_category = 'logout'

    print(f"{category_reciprocal_rank(predicted_categories, true_category):.2f}")
    print(f"{category_reciprocal_rank(predicted_categories, false_category):.2f}")