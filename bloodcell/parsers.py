"""
==============================================================
BloodCellAI Framework
Universal Annotation Parsers
==============================================================
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


# ==========================================================
# Universal Class Mapping
# ==========================================================

CLASS_MAP = {

    # RBC
    "RBC": 0,
    "Red Blood Cell": 0,
    "red blood cell": 0,

    # WBC
    "WBC": 1,
    "White Blood Cell": 1,
    "white blood cell": 1,
    "leukocyte": 1,

    # Platelet
    "Platelet": 2,
    "platelet": 2,
    "Platelets": 2,
    "platelets": 2,

    # Malaria Parasite
    "Parasite": 3,
    "parasite": 3,
    "ring": 3,
    "troph": 3,
    "trophozoite": 3,
    "schizont": 3,
    "merozoite": 3,
    "gametocyte": 3
}


# ==========================================================
# Detect Annotation Format
# ==========================================================

def detect_annotation_format(file_path):

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


# ==========================================================
# YOLO TXT Parser
# ==========================================================

def parse_yolo(txt_file):
    """
    Standard YOLO annotation parser.

    Returns
    -------
    [[class, xc, yc, w, h], ...]
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


# ==========================================================
# Pascal VOC XML Parser
# ==========================================================

def parse_pascal_xml(xml_file):
    """
    Parse Pascal VOC XML.

    Returns
    -------
    [[class, xc, yc, w, h], ...]
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    width = float(root.find("size/width").text)
    height = float(root.find("size/height").text)

    annotations = []

    for obj in root.findall("object"):

        cls_name = obj.find("name").text.strip()

        if cls_name not in CLASS_MAP:
            logger.warning(
                "parse_pascal_xml: class name %r in %s is not in "
                "CLASS_MAP and will be dropped -- add it if it "
                "represents a real class (this previously silently "
                "dropped all 'Platelets' objects in the real BCCD "
                "dataset before that variant was added).",
                cls_name, xml_file,
            )
            continue

        cls = CLASS_MAP[cls_name]

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        xc = ((xmin + xmax) / 2) / width
        yc = ((ymin + ymax) / 2) / height

        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        annotations.append([cls, xc, yc, w, h])

    return annotations


# ==========================================================
# Malaria Bounding Boxes JSON Parser
# ==========================================================

def parse_malaria_json(json_record):
    """
    Parse one record from the NIH Malaria Bounding Boxes dataset
    (https://data.lhncbc.nlm.nih.gov/public/Malaria/) JSON format:

        {
            "image": {"shape": {"r": <height>, "c": <width>, ...}, ...},
            "objects": [
                {
                    "category": "trophozoite" | "red blood cell" | ...,
                    "bounding_box": {
                        "minimum": {"r": <ymin>, "c": <xmin>},
                        "maximum": {"r": <ymax>, "c": <xmax>}
                    }
                },
                ...
            ]
        }

    Categories not present in CLASS_MAP are skipped rather than raising,
    since the raw dataset includes several fine-grained parasite life-cycle
    labels (ring/troph/schizont/gametocyte) that we intentionally collapse
    into a single "Parasite" class via CLASS_MAP.

    Returns
    -------
    [[class, xc, yc, w, h], ...]
        Same normalized [class, xc, yc, w, h] shape as the other parsers,
        so adapters.py can treat all datasets uniformly.
    """

    image_info = json_record.get("image", {})
    shape = image_info.get("shape", {})

    width = float(shape.get("c", 0)) or 1.0
    height = float(shape.get("r", 0)) or 1.0

    annotations = []

    for obj in json_record.get("objects", []):

        category = str(obj.get("category", "")).strip().lower()

        cls = CLASS_MAP.get(category)

        if cls is None:
            continue

        bbox = obj.get("bounding_box", {})

        minimum = bbox.get("minimum", {})
        maximum = bbox.get("maximum", {})

        xmin = float(minimum.get("c", 0))
        ymin = float(minimum.get("r", 0))
        xmax = float(maximum.get("c", 0))
        ymax = float(maximum.get("r", 0))

        xc = ((xmin + xmax) / 2) / width
        yc = ((ymin + ymax) / 2) / height

        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        annotations.append([cls, xc, yc, w, h])

    return annotations


# ==========================================================
# Chula RBC Parser
# ==========================================================

def parse_chula_txt(
    txt_file,
    image_width=640,
    image_height=480,
    default_box=70
):
    """
    Parse Chula RBC annotations.

    Annotation format
    -----------------
    x y morphology_class

    Returns
    -------
    [[class, xc, yc, w, h], ...]
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

            # IMPORTANT:
            # Keep the original Chula morphology class
            cls = int(parts[2])

            xc = x / image_width
            yc = y / image_height

            w = default_box / image_width
            h = default_box / image_height

            annotations.append([
                cls,
                xc,
                yc,
                w,
                h
            ])

    return annotations


# ==========================================================
# RBC Morphology Parser
# ==========================================================

def parse_rbc_morphology_txt(
    txt_file,
    image_width=640,
    image_height=480,
    default_box=70
):
    """
    Parse RBC Morphology dataset.

    Annotation format
    -----------------
    x y class

    Returns
    -------
    [[class, xc, yc, w, h], ...]
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

            annotations.append([
                cls,
                xc,
                yc,
                w,
                h
            ])

    return annotations

# ==========================================================
# CBC Report Parser (Clinical)
# ==========================================================

def parse_cbc_report(annotation_file):
    """
    Parse a plain-text CBC (Complete Blood Count) report.

    Expected format: one "Key: value" pair per line for numeric CBC
    fields, e.g.:

        Hemoglobin: 13.5
        RBC_Count: 4.8
        WBC_Count: 7200
        Platelet_Count: 250000
        MCV: 88
        MCH: 29
        MCHC: 33
        Morphology: Microcytic, Hypochromic

    The "Morphology:" line is intentionally skipped here -- it is
    parsed separately by extract_morphology_findings(), since it is a
    list of findings, not a numeric value.

    Returns
    -------
    dict[str, float]
        e.g. {"Hemoglobin": 13.5, "RBC_Count": 4.8, ...}
    """

    cbc = {}

    with open(annotation_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line or ":" not in line:
                continue

            key, _, value = line.partition(":")

            key = key.strip()
            value = value.strip()

            if key.lower() == "morphology":
                continue

            try:
                cbc[key] = float(value)
            except ValueError:
                logger.warning(
                    "parse_cbc_report: could not parse value for "
                    "%r in %s: %r",
                    key, annotation_file, value,
                )
                continue

    return cbc


def extract_morphology_findings(annotation_file):
    """
    Extract RBC morphology findings from a CBC report text file.

    Expects a line of the form "Morphology: Finding1, Finding2, ...".
    Returns [] if no such line is present (e.g. an unremarkable smear
    with nothing to report).

    Returns
    -------
    list[str]
        e.g. ["Microcytic", "Hypochromic"]
    """

    findings = []

    with open(annotation_file, "r") as f:

        for line in f:

            line = line.strip()

            if line.lower().startswith("morphology"):

                _, _, value = line.partition(":")

                findings = [
                    item.strip()
                    for item in value.split(",")
                    if item.strip()
                ]

    return findings


# ==========================================================
# COCO JSON Parser
# ==========================================================

def parse_coco_json(data: dict):
    """
    Parse a full COCO-format annotation dict (one JSON file
    describes the ENTIRE dataset, unlike per-image YOLO/VOC files):

        {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
                ...
            ],
            "annotations": [
                {"image_id": 1, "category_id": 2,
                 "bbox": [x, y, w, h], ...},   # absolute pixels, top-left corner
                ...
            ],
            "categories": [
                {"id": 2, "name": "RBC"},
                ...
            ]
        }

    Since one COCO file covers every image, this is parsed once and
    indexed by filename, rather than being re-parsed per image the
    way per-image parsers work (see CocoAdapter, which caches this
    result rather than calling it once per image).

    Returns
    -------
    tuple
        (images_by_filename, categories) where:
        - images_by_filename: dict[str, dict] -- keyed by the image's
          file_name, each value has "width", "height", "annotations"
          (the list of raw COCO annotation dicts for that image).
        - categories: dict[int, str] -- category_id -> category name.
    """

    categories = {
        category["id"]: category.get("name", f"Class_{category['id']}")
        for category in data.get("categories", [])
    }

    images_by_id = {}

    for image in data.get("images", []):

        images_by_id[image["id"]] = {
            "file_name": image.get("file_name", ""),
            "width": image.get("width", 0),
            "height": image.get("height", 0),
            "annotations": [],
        }

    for annotation in data.get("annotations", []):

        image_id = annotation.get("image_id")

        if image_id in images_by_id:
            images_by_id[image_id]["annotations"].append(annotation)

    images_by_filename = {
        record["file_name"]: record
        for record in images_by_id.values()
        if record["file_name"]
    }

    return images_by_filename, categories
