def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    level = series[0]
    trend = series[1] - series[0]
    result = [level]

    for y in series[1:]:
        old_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - old_level) + (1 - beta) * trend
        result.append(level)

    return result