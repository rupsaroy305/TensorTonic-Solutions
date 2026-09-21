def lag_features(series: list, lags: list) -> list:
    start = max(lags)
    return [[series[t - lag] for lag in lags]
            for t in range(start, len(series))]