import numpy as np

def mean_median_mode(x):
    x = np.asarray(x, dtype=float)

    mean = float(np.mean(x))
    median = float(np.median(x))

    values, counts = np.unique(x, return_counts=True)
    mode = float(values[np.argmax(counts)])  
    return mean, median, mode