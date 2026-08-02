
"""
BloodCellAI Framework
Universal Annotation Parsers
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_MAP = {
    "RBC": 0,
    "Red Blood Cell": 0,
    "red blood cell": 0,

    "WBC": 1,
    "White Blood Cell": 1,
    "white blood cell": 1,

    "Platelet": 2,
    "platelet": 2,

    "Parasite": 3,
    "parasite": 3,
    "ring": 3,
    "troph": 3,
    "trophozoite": 3,
    "schizont": 3,
    "merozoite": 3,
    "gametocyte": 3
}


def parse_yolo(txt_file):
    """
    Standard YOLO parser
    Returns:
        [[class,x,y,w,h], ...]
    """
    annotations = []

    with open(txt_file, "r") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            values = list(map(float, line.split()))

            annotations.append(values)

    return annotations


def detect_annotation_format(file_path):
    """
    Detect annotation type from extension.
    """

    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        return "TXT"

    if ext == ".xml":
        return "XML"

    if ext == ".json":
        return "JSON"

    if ext == ".csv":
        return "CSV"

    return "UNKNOWN"


def parse_pascal_xml(xml_file):
    """
    Parse Pascal VOC XML annotations.

    Returns
    -------
    list
        [[class,x,y,w,h], ...] in YOLO normalized format
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    width = float(root.find("size/width").text)
    height = float(root.find("size/height").text)

    annotations = []

    for obj in root.findall("object"):

        cls_name = obj.find("name").text.strip()

        if cls_name not in CLASS_MAP:
            continue

        cls = CLASS_MAP[cls_name]

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x = ((xmin + xmax) / 2) / width
        y = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        annotations.append([cls, x, y, w, h])

    return annotations



def parse_chula_txt(txt_file, image_width=640, image_height=480):
    """
    Parse Chula RBC TXT annotation.

    Format:
        x y class

    Returns
    -------
    [[class,x,y,w,h], ...]
    """

    annotations = []

    DEFAULT_BOX = 70

    with open(txt_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            x = float(parts[0])
            y = float(parts[1])

            cls = int(parts[2])

            xc = x / image_width
            yc = y / image_height

            w = DEFAULT_BOX / image_width
            h = DEFAULT_BOX / image_height

            annotations.append([cls, xc, yc, w, h])

    return annotations



def parse_rbc_morphology_txt(
    txt_file,
    image_width=640,
    image_height=480,
    default_box=70
):
    """
    Parse RBC Morphology TXT annotations.

    Expected format:
        x y class

    Returns
    -------
    [[class,x,y,w,h], ...]
    """

    annotations = []

    with open(txt_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            x = float(parts[0])
            y = float(parts[1])
            cls = int(parts[2])

            xc = x / image_width
            yc = y / image_height

            w = default_box / image_width
            h = default_box / image_height

            annotations.append([cls, xc, yc, w, h])

    return annotations

