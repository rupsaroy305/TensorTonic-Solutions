import numpy as np

def apply_homogeneous_transform(T, points):
    points = np.asarray(points)
    single = points.ndim == 1
    if single:
        points = points[None, :]
    h = np.c_[points, np.ones(len(points))]
    out = (T @ h.T).T[:, :3]
    return out[0] if single else out