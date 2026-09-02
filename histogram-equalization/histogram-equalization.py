import numpy as np
def histogram_equalize(image: list) -> list:
    img = np.array(image)
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = np.cumsum(hist)

    cdf_min = cdf[cdf > 0][0]
    n = img.size

    if n == cdf_min:
        return np.zeros_like(img).tolist()

    mapping = np.round((cdf - cdf_min) / (n - cdf_min) * 255).astype(int)

    return mapping[img].tolist()