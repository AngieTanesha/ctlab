"""Misc
=======

This file contains various functions which do not fit anywhere else in
particular and might be helpful.

"""
import numpy as np
from PIL import Image


def array_to_img(data):
    """Converts the 2D array into an image.

    The maximum value within the array is mapped to white and the minimum to
    black.

    """
    max_val = data.max()
    min_val = data.min()
    diff = max_val - min_val

    img = Image.fromarray((255 * (data - min_val) / diff).astype(np.uint8),mode='L',)

    return img
