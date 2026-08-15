import numpy as np
def log_transform(values):
    values=np.asarray(values,dtype=float)
    return np.log1p(values).tolist()