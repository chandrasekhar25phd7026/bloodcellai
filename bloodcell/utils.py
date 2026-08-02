
from pathlib import Path
import hashlib
from PIL import Image


def create_folder(folder):
    """
    Create folder if it does not exist.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def image_size(image_path):
    """
    Returns image width and height.
    """
    img = Image.open(image_path)
    return img.size


def file_md5(file_path):
    """
    Compute MD5 checksum of a file.
    """
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            md5.update(chunk)

    return md5.hexdigest()


def is_image(file_path):
    """
    Check whether file is an image.
    """
    ext = Path(file_path).suffix.lower()

    return ext in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    ]
