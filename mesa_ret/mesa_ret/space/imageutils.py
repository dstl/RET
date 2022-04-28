"""Image utilities for spaces."""

import math
import warnings

import numpy as np


def check_image_aspect_ratio(
    image_shape: tuple[int, int], image_name: str, expected_shape: tuple[float, float]
) -> None:
    """Check the aspect ratio of the given image matches the expected shape to within 5%.

    Args:
        image_shape (tuple[int, int]): tuple of x and y size of the image to check.
        image_name (str): Name of the image used for the warning message.
        expected_shape (tuple[float, float]): tuple of x and y size of the expected
            shape.

    """
    image_aspect_ratio = image_shape[1] / image_shape[0]
    expected_aspect_ratio = expected_shape[1] / expected_shape[0]
    if not math.isclose(image_aspect_ratio, expected_aspect_ratio, rel_tol=5e-2):
        warnings.warn(
            f"{image_name} image aspect ratio differs from the space aspect ratio "
            + "by more than 5%, the image will be stretched to span the space"
        )


def get_image_pixel_size(
    image: np.ndarray, x_distance: float, y_distance: float
) -> tuple[float, float, float]:
    """Return smallest pixel size in x or y axis, in units of model space distance.

    Args:
        image (np.array): image describing model space
        x_distance (float): x dimension of entire image
        y_distance (float): y dimension of entire image

    Returns:
        tuple[float, float, float]: (smallest pixel size in x or y direction,
            multiplication factor between number of x pixels and x distance,
            multiplication factor between number of y pixels and y distance)
    """
    x_pixels = image.shape[0]
    y_pixels = image.shape[1]

    x_factor = x_pixels / x_distance
    y_factor = y_pixels / y_distance

    x_pixel_size = 1 / x_factor
    y_pixel_size = 1 / y_factor

    return min(x_pixel_size, y_pixel_size), x_factor, y_factor
