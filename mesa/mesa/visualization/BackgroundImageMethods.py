import math
import os
import pathlib
import warnings
from shutil import copy

from PIL import Image


def find_background_image_path(canvas_width, canvas_height, canvas_background_file):
    """Establish the path to the background file should one exist. The function copies it from wherever specified into the root.
        Also checks the aspect ratio of the input file against the canvas.

    Args:
        canvas_height, canvas_width: The width and height selected for the canvas.
        canvas_background_file: Absolute path to the image.

    """
    if canvas_background_file:
        check_image_aspect_ratio(canvas_width, canvas_height, canvas_background_file)
        # copy canvas_background_file to mesa\visualization\images\background_image
        background_file_extension = pathlib.Path(canvas_background_file).suffix
        pathlib.Path(os.path.join(os.path.dirname(__file__), "./images")).mkdir(
            parents=True, exist_ok=True
        )
        copy(
            canvas_background_file,
            os.path.join(
                os.path.dirname(__file__),
                "./images/background_image" + background_file_extension,
            ),
        )
        canvas_background_path = "/assets/background_image" + background_file_extension
    else:
        canvas_background_path = ""
    return canvas_background_path


def check_image_aspect_ratio(canvas_width, canvas_height, image_file_path: str):
    """Check the aspect ratio of the given shape matches the aspect ratio of the space
    to within 5%.

    Args:
        canvas_height, canvas_width: The width and height selected for the canvas.
        canvas_background_file: Absolute path to the image.

    """
    image = Image.open(image_file_path)
    width, height = image.size
    image_aspect_ratio = width / height
    space_aspect_ratio = canvas_width / canvas_height
    if not math.isclose(image_aspect_ratio, space_aspect_ratio, rel_tol=5e-2):
        warnings.warn(
            pathlib.Path(image_file_path).name
            + " image aspect ratio differs from the space aspect ratio by more than 5%, the image will be stretched to span the space"
        )
