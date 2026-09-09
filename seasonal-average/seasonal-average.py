import numpy as np
def seasonal_average(series: list, period: int) -> list:
    result=[]
    for p in range(period):
        result.append(np.mean(series[p::period]))
    return result