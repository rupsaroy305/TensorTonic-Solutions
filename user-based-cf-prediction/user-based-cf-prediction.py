def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    weighted_sum = 0
    similarity_sum = 0

    for s, r in zip(similarities, ratings):
        if s > 0:
            weighted_sum += s * r
            similarity_sum += s

    if similarity_sum == 0:
        return 0.0

    return round(weighted_sum / similarity_sum, 6)