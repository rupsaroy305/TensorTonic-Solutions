import numpy as np
def adjusted_cosine_similarity(ratings_matrix: list, item_i: int, item_j: int) -> float:
    matrix = np.array(ratings_matrix, dtype=float)

    numerator = 0.0
    sum_i = 0.0
    sum_j = 0.0

    for row in matrix:
        if row[item_i] != 0 and row[item_j] != 0:
            mean = np.mean(row[row != 0])

            a = row[item_i] - mean
            b = row[item_j] - mean

            numerator += a * b
            sum_i += a ** 2
            sum_j += b ** 2

    denominator = np.sqrt(sum_i * sum_j)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)