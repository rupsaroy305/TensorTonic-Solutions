import numpy as np
def autocorrelation(series: list, max_lag: int) -> list:
    x=np.array(series,dtype=float)
    x=x-np.mean(x)
    variance=np.sum(x**2)
    if variance==0:
        return [1.0]+[0.0]*max_lag
    result=[]
    for k in range(max_lag+1):
        result.append(round(float(np.sum(x[:len(x)-k]*x[k:])/variance),6))
    return result