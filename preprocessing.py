import numpy as np
from functools import reduce

def binarize_and_reshape(image: np.ndarray, shape: tuple):
    images_shape = image.shape
    desired_shape = images_shape[-1:0:-1]
    result = reduce(lambda a, b: a*b, desired_shape)

    reshaped_image = np.where(image.reshape((image.shape[0], result)) > 75, 1, 0)

    return reshaped_image