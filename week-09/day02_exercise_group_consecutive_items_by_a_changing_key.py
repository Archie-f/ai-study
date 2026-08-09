from itertools import groupby

def group_consecutive_items(flat_list: list[tuple[str | None, int]])-> list[list[str | None]]:
    """Groups consecutive items in flat list.

        Args:
            flat_list (list[tuple[str | None, int]]): List of tuples
        Returns:
            list[list[str | None]]: List of lists
    """
    groups: list[list[str | None]] = []
    for group_id, group in groupby(flat_list, key=lambda x: x[1]):
        current_group = [item[0] for item in group]
        groups.append(current_group)
    return groups