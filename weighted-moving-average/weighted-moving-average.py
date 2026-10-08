def weighted_moving_average(values: list, weights: list) -> list:
    k = len(weights)
    total = sum(weights)
    return [
        sum(values[i+j] * weights[j] for j in range(k)) / total
        for i in range(len(values) - k + 1)
    ]