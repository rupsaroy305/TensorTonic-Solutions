def catalog_coverage(recommendations: list, n_items: int) -> float:
    if n_items==0:
        return 0.0
    unique_items=set()
    for items in recommendations:
        unique_items.update(items)
    return float(len(unique_items)/n_items)