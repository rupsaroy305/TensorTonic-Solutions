def expected_calibration_error(y_true, y_pred, n_bins):
    import numpy as np
    y,p=np.array(y_true),np.array(y_pred)
    b=np.minimum((p*n_bins).astype(int),n_bins-1)
    e=0
    for i in range(n_bins):
        m=b==i
        if m.any():e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(e)