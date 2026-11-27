import numpy as np

def random_crop(image: np.ndarray, crop_size: int,
                crop_y: int, crop_x: int) -> np.ndarray:
    image=np.array(image,dtype=np.float64)
    return image[crop_y:crop_y+crop_size,crop_x:crop_x+crop_size,:].copy()


def random_horizontal_flip(image: np.ndarray, p: float,
                           flip_rand: float) -> np.ndarray:
    image=np.array(image,dtype=np.float64)
    if flip_rand<p:
        return image[:,::-1,:].copy()
    return image.copy()