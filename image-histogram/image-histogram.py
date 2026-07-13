import numpy as np

def image_histogram(image):
    image=np.asarray(image,dtype=int)
    return np.bincount(image.ravel(),minlength=256).tolist()