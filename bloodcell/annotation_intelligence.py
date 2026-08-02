
"""
BloodCellAI
Annotation Intelligence Engine v3

Detect annotation formats using content inspection.
"""

from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET
import re


# ----------------------------------------------------------
# TXT DETECTORS
# ----------------------------------------------------------

def is_yolo(lines):

    for line in lines[:20]:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:
            values = list(map(float, parts))

            if values[0] == int(values[0]):

                return True

        except Exception:
            pass

    return False


def is_chula(lines):

    for line in lines[:20]:

        parts = line.strip().split()

        if len(parts) != 3:
            continue

        try:
            list(map(int, parts))
            return True

        except Exception:
            pass

    return False


def is_cbc(text):

    return (
        "wbc," in text
        and "rbc," in text
        and "hgb" in text
    )


def is_morphology(text):

    keywords = [

        "peripheral smear",
        "rbc morphology",
        "platelets",
        "hypochromic",
        "microcytic"

    ]

    score = sum(k in text for k in keywords)

    return score >= 2


# ----------------------------------------------------------
# XML
# ----------------------------------------------------------

def is_pascal_xml(file):

    try:

        root = ET.parse(file).getroot()

        return root.tag == "annotation"

    except Exception:

        return False


# ----------------------------------------------------------
# JSON
# ----------------------------------------------------------

def is_malaria_json(file):

    try:

        with open(file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):

            first = data[0]

            return (
                "image" in first
                and "objects" in first
            )

    except Exception:

        pass

    return False


def is_coco_json(file):
    """
    COCO format is a single JSON *object* (not a list, unlike Malaria
    JSON) with top-level "images", "annotations", and "categories"
    keys.
    """

    try:

        with open(file, "r") as f:
            data = json.load(f)

        return (
            isinstance(data, dict)
            and "images" in data
            and "annotations" in data
            and "categories" in data
        )

    except Exception:

        pass

    return False


# ----------------------------------------------------------
# MASTER DETECTOR
# ----------------------------------------------------------

def detect_annotation(file):

    file = Path(file)

    suffix = file.suffix.lower()

    # ----------------------
    # TXT
    # ----------------------

    if suffix == ".txt":

        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).lower()

        lines = text.splitlines()

        if is_yolo(lines):
            return "YOLO"

        if is_chula(lines):
            return "Chula TXT"

        if is_cbc(text):
            return "CBC Report"

        if is_morphology(text):
            return "Morphology Report"

        return "Generic TXT"

    # ----------------------
    # JSON
    # ----------------------

    if suffix == ".json":

        if is_malaria_json(file):
            return "Malaria JSON"

        if is_coco_json(file):
            return "COCO"

        return "JSON"

    # ----------------------
    # XML
    # ----------------------

    if suffix == ".xml":

        if is_pascal_xml(file):
            return "Pascal XML"

        return "XML"

    # ----------------------
    # CSV
    # ----------------------

    if suffix == ".csv":
        return "CSV"

    return "Unknown"


# ----------------------------------------------------------
# DATASET-LEVEL (WHOLE-FOLDER) FORMAT DETECTION
# ----------------------------------------------------------
#
# Everything above this point answers "what format is THIS ONE
# annotation file in?" -- useful once you already know an image has
# a matching annotation file to inspect. It does not answer "what
# format is this whole folder in?", and nothing in the package wired
# it up to actually look at a folder. detect_dataset_format() below
# is the missing piece: point it at an arbitrary directory and it
# guesses task + annotation format + evidence, without requiring a
# prior DATASET_REGISTRY entry -- this is what actually enables
# "point BloodCellAI at a folder and have it auto-detect the format."

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_yaml_like(file):
    """
    Minimal parser for the flat data.yaml / data.yml manifests used
    by Roboflow / YOLOv5 / YOLOv8 exports, e.g.:

        train: ../train/images
        val: ../valid/images
        test: ../test/images
        nc: 3
        names: ['RBC', 'WBC', 'Platelet']

    Uses PyYAML if it's installed (handles edge cases properly);
    falls back to a small manual parser for this common flat
    key: value / key: [list] shape if PyYAML isn't available, so this
    doesn't force a hard dependency just for a handful of well-known
    keys.
    """

    try:
        import yaml

        with open(file, "r") as f:
            return yaml.safe_load(f) or {}

    except ImportError:
        pass
    except Exception:
        return {}

    result = {}

    try:

        with open(file, "r") as f:
            text = f.read()

        for line in text.splitlines():

            line = line.split("#", 1)[0].strip()

            if not line or ":" not in line:
                continue

            key, _, value = line.partition(":")

            key = key.strip()
            value = value.strip()

            if value.startswith("[") and value.endswith("]"):

                items = [
                    item.strip().strip("'\"")
                    for item in value[1:-1].split(",")
                    if item.strip()
                ]

                result[key] = items

            elif value.isdigit():

                result[key] = int(value)

            else:

                result[key] = value.strip("'\"")

    except Exception:

        return {}

    return result


def _find_roboflow_manifest(dataset_path):
    """
    Look for a data.yaml / data.yml at the dataset root, the
    convention used by Roboflow and YOLOv5/v8 exports.
    """

    for name in ("data.yaml", "data.yml"):

        candidate = dataset_path / name

        if candidate.exists():
            return candidate

    return None


def _looks_like_folder_per_class(dataset_path):
    """
    Detects the classification convention used by LISC / NIH_Malaria
    / ALL_IDB / etc: images grouped into one subfolder per class,
    with no separate annotation files anywhere.
    """

    subdirs = [p for p in dataset_path.iterdir() if p.is_dir()]

    if not subdirs:
        return False, {}

    class_names = {}

    for i, subdir in enumerate(sorted(subdirs)):

        images_here = any(
            f.suffix.lower() in IMAGE_EXTENSIONS
            for f in subdir.rglob("*")
            if f.is_file()
        )

        if images_here:
            class_names[i] = subdir.name

    return bool(class_names), class_names


def _extract_voc_class_names(xml_files, max_files=50):
    """
    Extract real class names from a sample of Pascal VOC XML files'
    <object><name>...</name></object> tags.

    IMPORTANT: ids are looked up in the same global CLASS_MAP that
    parsers.parse_pascal_xml() actually uses to assign class_id
    (RBC=0, WBC=1, Platelet=2, ...) -- NOT assigned by first-appearance
    order. Assigning by appearance order was tried first and is
    wrong: it silently mismatches whatever id parse_pascal_xml
    actually gives each object, since that id comes from CLASS_MAP,
    not from the order names happen to appear in a sampled subset of
    files. Confirmed on real BCCD data: appearance-order assignment
    swapped the WBC and Platelet labels because "Platelets" happened
    to appear before "WBC" in the sampled files.

    A name not present in CLASS_MAP is skipped here, matching
    parse_pascal_xml's own behavior of silently dropping (with a
    logged warning) any object whose class name isn't recognized --
    such objects will never appear in the built dataset regardless of
    what this function reports, so there's no id to assign them here.
    """

    from .parsers import CLASS_MAP

    names = {}

    for xml_file in xml_files[:max_files]:

        try:
            root = ET.parse(xml_file).getroot()
        except Exception:
            continue

        for obj in root.findall("object"):

            name_tag = obj.find("name")

            if name_tag is None or not name_tag.text:
                continue

            name = name_tag.text.strip()

            class_id = CLASS_MAP.get(name)

            if class_id is not None and class_id not in names:
                names[class_id] = name

    return dict(sorted(names.items()))


def detect_dataset_format(dataset_path):
    """
    Inspect an arbitrary folder and guess its dataset format and
    task, without requiring a prior DATASET_REGISTRY entry.

    Parameters
    ----------
    dataset_path : str or Path

    Returns
    -------
    dict
        {
            "task": "Detection" | "Classification" | "Unknown",
            "annotation": "YOLO" | "COCO" | "Pascal VOC" | "None" | "Unknown",
            "image_extension": str or None,
            "label_extension": str or None,
            "is_roboflow_export": bool,
            "classes": dict[int, str]  (best-effort; may be empty),
            "evidence": list[str]  (human-readable reasons for the decision),
        }
    """

    dataset_path = Path(dataset_path)

    evidence = []

    result = {
        "task": "Unknown",
        "annotation": "Unknown",
        "image_extension": None,
        "label_extension": None,
        "is_roboflow_export": False,
        "classes": {},
        "evidence": evidence,
    }

    if not dataset_path.is_dir():
        evidence.append(f"{dataset_path} is not a directory.")
        return result

    # ------------------------------------------------------------
    # Roboflow / YOLOv5-v8 manifest (data.yaml) -- checked first,
    # since it directly names the classes and is unambiguous when
    # present, rather than relying on statistical guesses.
    # ------------------------------------------------------------

    manifest = _find_roboflow_manifest(dataset_path)

    if manifest is not None:

        result["is_roboflow_export"] = True
        evidence.append(f"Found Roboflow/YOLO manifest: {manifest.name}")

        manifest_data = _read_yaml_like(manifest)

        names = manifest_data.get("names")

        if isinstance(names, list):
            result["classes"] = {i: name for i, name in enumerate(names)}
        elif isinstance(names, dict):
            result["classes"] = {int(k): v for k, v in names.items()}

    # ------------------------------------------------------------
    # Collect a sample of files across the whole tree (Roboflow
    # exports commonly nest images under train/valid/test
    # subfolders, so this must be recursive, not just the top level).
    # ------------------------------------------------------------

    all_files = [f for f in dataset_path.rglob("*") if f.is_file()]

    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTENSIONS]
    json_files = [f for f in all_files if f.suffix.lower() == ".json"]
    xml_files = [f for f in all_files if f.suffix.lower() == ".xml"]
    txt_files = [
        f for f in all_files
        if f.suffix.lower() == ".txt" and f.name.lower() not in ("classes.txt", "readme.txt")
    ]
    csv_files = [f for f in all_files if f.suffix.lower() == ".csv"]

    if image_files:
        result["image_extension"] = Counter(
            f.suffix.lower() for f in image_files
        ).most_common(1)[0][0]

    # ------------------------------------------------------------
    # COCO: one (or a few, e.g. per-split) JSON files shaped like COCO.
    # ------------------------------------------------------------

    coco_candidates = [f for f in json_files if is_coco_json(f)]

    if coco_candidates:

        result["task"] = "Detection"
        result["annotation"] = "COCO"
        result["label_extension"] = ".json"

        evidence.append(
            f"Found {len(coco_candidates)} COCO-shaped JSON file(s): "
            f"{[f.name for f in coco_candidates[:3]]}"
        )

        return result

    # ------------------------------------------------------------
    # Pascal VOC: multiple XML files with <annotation> root.
    # ------------------------------------------------------------

    if xml_files:

        sample = xml_files[:20]
        voc_hits = sum(1 for f in sample if is_pascal_xml(f))

        if voc_hits >= max(1, len(sample) // 2):

            result["task"] = "Detection"
            result["annotation"] = "Pascal VOC"
            result["label_extension"] = ".xml"

            evidence.append(
                f"{voc_hits}/{len(sample)} sampled XML files have a "
                "Pascal VOC <annotation> root."
            )

            result["classes"] = _extract_voc_class_names(sample)

            if result["classes"]:
                evidence.append(
                    f"Extracted {len(result['classes'])} class name(s) "
                    f"from <name> tags: {list(result['classes'].values())}"
                )

            return result

    # ------------------------------------------------------------
    # YOLO: multiple TXT files, each a handful of "class x y w h"
    # numeric lines.
    # ------------------------------------------------------------

    if txt_files:

        sample = txt_files[:20]

        yolo_hits = 0

        for f in sample:

            try:
                lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue

            if is_yolo(lines):
                yolo_hits += 1

        if yolo_hits >= max(1, len(sample) // 2):

            result["task"] = "Detection"
            result["annotation"] = "YOLO"
            result["label_extension"] = ".txt"

            evidence.append(
                f"{yolo_hits}/{len(sample)} sampled TXT files match the "
                "YOLO 'class xc yc w h' shape."
            )

            return result

    # ------------------------------------------------------------
    # CSV-labeled classification: a dataset-wide CSV plus images,
    # no per-image annotation files.
    # ------------------------------------------------------------

    if csv_files and image_files and not txt_files and not xml_files:

        result["task"] = "Classification"
        result["annotation"] = "CSV"
        result["label_extension"] = ".csv"

        evidence.append(
            f"Found {len(csv_files)} CSV file(s) alongside images and "
            "no per-image annotation files."
        )

        return result

    # ------------------------------------------------------------
    # Folder-per-class classification: no annotation files anywhere,
    # images grouped into class-named subdirectories.
    # ------------------------------------------------------------

    if not txt_files and not xml_files and not json_files and not csv_files:

        is_folder_per_class, class_names = _looks_like_folder_per_class(dataset_path)

        if is_folder_per_class:

            result["task"] = "Classification"
            result["annotation"] = "None"

            if class_names and not result["classes"]:
                result["classes"] = class_names

            evidence.append(
                f"No annotation files found; images are grouped into "
                f"{len(class_names)} class-named subfolders: "
                f"{list(class_names.values())[:5]}"
            )

            return result

    evidence.append(
        "Could not confidently match any known format "
        "(YOLO / COCO / Pascal VOC / CSV / folder-per-class)."
    )

    return result
