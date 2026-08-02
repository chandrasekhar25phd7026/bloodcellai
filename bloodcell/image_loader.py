
"""
==============================================================
BloodCellAI Universal Image Loader
==============================================================
"""

from pathlib import Path
from PIL import Image


def load_image_info(image_path):

    image_path = Path(image_path)

    with Image.open(image_path) as img:

        width, height = img.size

    return {

        "image_path": str(image_path),

        "width": width,

        "height": height

    }

