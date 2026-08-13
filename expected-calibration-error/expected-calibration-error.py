import numpy as np
def expected_calibration_error(y_true, y_pred, n_bins):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    bin_indices = np.floor(y_pred * n_bins).astype(int)
    bin_indices = np.minimum(bin_indices, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_y = y_true[mask]
            bin_p = y_pred[mask]
            accuracy = np.mean(bin_y)
            confidence = np.mean(bin_p)
            weight = np.sum(mask) / n
            ece += weight * abs(accuracy - confidence)
    return float(ece)