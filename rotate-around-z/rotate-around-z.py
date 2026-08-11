import numpy as np

def rotate_around_z(points, theta):
    points = np.asarray(points, dtype=float)

    c = np.cos(theta)
    s = np.sin(theta)

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    x_new = x * c - y * s
    y_new = x * s + y * c

    return np.stack([x_new, y_new, z], axis=-1)