
"""
==============================================================
BloodCellAI Dataset Scanner
==============================================================
"""

from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",".jpeg",".png",".bmp",".tif",".tiff"
}

ANNOTATION_EXTENSIONS = {
    ".txt",".xml",".json",".csv"
}


def scan_dataset(dataset):

    dataset.image_files = []

    dataset.annotation_files = []

    for file in Path(dataset.path).rglob("*"):

        if not file.is_file():
            continue

        ext = file.suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            dataset.image_files.append(file)

        elif ext in ANNOTATION_EXTENSIONS:
            dataset.annotation_files.append(file)

    dataset.image_count = len(dataset.image_files)

    dataset.annotation_count = len(dataset.annotation_files)

    return dataset
