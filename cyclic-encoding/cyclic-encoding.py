import numpy as np
import math

def cyclic_encoding(values: list, period: float) -> list:
    values=np.array(values)
    theta=2*np.pi*values/period
    return np.column_stack((np.sin(theta),np.cos(theta))).tolist()