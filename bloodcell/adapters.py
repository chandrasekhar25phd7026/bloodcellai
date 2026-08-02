
from .parsers import *
from .universal_object import *
from .dataset_registry import get_classes

import json
import xml.etree.ElementTree as ET


# ============================================================
# Base Adapter
# ============================================================

class BaseAdapter:

    def convert(self, *args, **kwargs):
        raise NotImplementedError


# ============================================================
# Classification Adapter
# ============================================================

class ClassificationAdapter(BaseAdapter):
    """
    Generic adapter for whole-image classification datasets (e.g. LISC,
    NIH_Malaria, ALL_IDB, AcuteLeukemia): one label for the entire
    image, not per-cell bounding boxes.

    There is no natural "BoundingBox" for a classification label, but
    rather than inventing a parallel data model, this represents the
    whole-image label as a single BoundingBox spanning the full image
    (xc=yc=0.5, w=h=1.0). This lets classification datasets flow
    through the exact same UniversalImage / DatasetStatistics /
    DatasetValidator / BDQI machinery that every detection dataset
    already uses, with zero special-casing downstream -- the tradeoff
    is that "objects per image" and similar detection-oriented
    statistics are not meaningful for classification datasets and
    should be read as "1" by construction, not as a real object count.
    """

    def convert(
        self,
        image_path,
        class_id,
        dataset,
        width=0,
        height=0,
    ):

        classes = get_classes(dataset)

        img = UniversalImage(
            image_path=image_path,
            dataset=dataset,
            width=width,
            height=height,
        )

        img.objects.append(

            BoundingBox(

                class_id=int(class_id),

                class_name=classes.get(
                    int(class_id),
                    f"Class_{int(class_id)}"
                ),

                xc=0.5,
                yc=0.5,
                w=1.0,
                h=1.0,

                confidence=1.0,

                source=dataset

            )

        )

        return img


# ============================================================
# Pascal VOC Adapter
# ============================================================

class PascalVOCAdapter(BaseAdapter):
    """
    Adapter for datasets annotated in Pascal VOC XML format -- this is
    the format the real, canonical public BCCD dataset actually ships
    in (Annotations/*.xml), not YOLO .txt as the registry originally
    assumed.
    """

    def convert(
        self,
        annotation_file,
        image_path="",
        dataset="BCCD",
    ):

        tree = ET.parse(annotation_file)
        root = tree.getroot()

        width = int(float(root.find("size/width").text))
        height = int(float(root.find("size/height").text))

        anns = parse_pascal_xml(annotation_file)

        classes = get_classes(dataset)

        img = UniversalImage(
            image_path=image_path,
            dataset=dataset,
            width=width,
            height=height
        )

        for obj in anns:

            img.objects.append(

                BoundingBox(

                    class_id=int(obj[0]),

                    class_name=classes.get(
                        int(obj[0]),
                        f"Class_{int(obj[0])}"
                    ),

                    xc=obj[1],

                    yc=obj[2],

                    w=obj[3],

                    h=obj[4],

                    confidence=1.0,

                    source=dataset

                )

            )

        return img


# ============================================================
# YOLO Adapter
# ============================================================

class YOLOAdapter(BaseAdapter):

    def convert(
        self,
        annotation_file,
        image_path="",
        dataset="YOLO",
        width=640,
        height=640
    ):

        anns = parse_yolo(annotation_file)

        classes = get_classes(dataset)

        img = UniversalImage(
            image_path=image_path,
            dataset=dataset,
            width=width,
            height=height
        )

        for obj in anns:

            img.objects.append(

                BoundingBox(

                    class_id=int(obj[0]),

                    class_name=classes.get(
                        int(obj[0]),
                        f"Class_{int(obj[0])}"
                    ),

                    xc=obj[1],

                    yc=obj[2],

                    w=obj[3],

                    h=obj[4],

                    confidence=1.0,

                    source=dataset

                )

            )

        return img


# ============================================================
# Chula Adapter
# ============================================================

class ChulaAdapter(BaseAdapter):

    def convert(
        self,
        annotation_file,
        image_path="",
        dataset="Chula_RBC",
        width=640,
        height=480
    ):

        anns = parse_chula_txt(annotation_file)

        classes = get_classes(dataset)

        img = UniversalImage(
            image_path=image_path,
            dataset=dataset,
            width=width,
            height=height
        )

        for obj in anns:

            img.objects.append(

                BoundingBox(

                    class_id=int(obj[0]),

                    class_name=classes.get(
                        int(obj[0]),
                        f"Class_{int(obj[0])}"
                    ),

                    xc=obj[1],

                    yc=obj[2],

                    w=obj[3],

                    h=obj[4],

                    confidence=1.0,

                    source=dataset

                )

            )

        return img


# ============================================================
# Malaria Adapter
# ============================================================

class MalariaAdapter(BaseAdapter):
    """
    Adapter for the NIH/Kaggle Malaria Bounding Boxes JSON format.

    The real public dataset ships as a JSON *list* of per-image
    records in one or two shared files (e.g. training.json,
    test.json) -- confirmed from the actual dataset structure, not a
    single record per file. This adapter now handles that real shape
    (matching annotation_intelligence.is_malaria_json()'s detection,
    which also expects a list), while still tolerating a bare single
    record for backward compatibility with any per-image-file setup.

    Note: this previously took a pre-parsed `json_record` dict as its
    first argument, which does not match how build_single_image() in
    pipeline.py actually calls every adapter (it always passes
    `annotation_file=<path>`). That meant this dataset could never be
    built through UniversalBuilder -- only by calling the adapter
    directly. Fixed to accept `annotation_file` like every other
    adapter and parse the JSON itself. It also previously assumed the
    parsed JSON was always a single record (`json_record["image"]`),
    which breaks immediately on the real multi-record file shape --
    confirmed by actually building a real-shaped test file, not just
    a single-image synthetic one.
    """

    def __init__(self):

        self._cached_path = None
        self._cached_records_by_filename = None

    def _load(self, annotation_file):

        annotation_file = str(annotation_file)

        if self._cached_path == annotation_file:
            return self._cached_records_by_filename

        with open(annotation_file, "r") as f:
            data = json.load(f)

        records = data if isinstance(data, list) else [data]

        records_by_filename = {}

        for record in records:

            image_info = record.get("image", {})

            pathname = image_info.get("pathname", "") or image_info.get("file_name", "")

            filename = Path(pathname).name if pathname else None

            if filename:
                records_by_filename[filename] = record

        self._cached_path = annotation_file
        self._cached_records_by_filename = records_by_filename

        return records_by_filename

    def convert(
        self,
        annotation_file,
        image_path="",
        dataset="Malaria Bounding Boxes",
    ):

        records_by_filename = self._load(annotation_file)

        filename = Path(image_path).name

        record = records_by_filename.get(filename)

        if record is None:
            # Fall back to stem matching, same tolerance as CocoAdapter,
            # in case of extension mismatches between the JSON record
            # and the actual file on disk.
            stem = Path(image_path).stem
            for candidate_name, candidate_record in records_by_filename.items():
                if Path(candidate_name).stem == stem:
                    record = candidate_record
                    break

        classes = get_classes(dataset)

        if record is None:
            return UniversalImage(
                image_path=image_path, dataset=dataset, width=0, height=0
            )

        anns = parse_malaria_json(record)

        image_info = record.get("image", {})
        width = image_info.get("shape", {}).get("c", 0)
        height = image_info.get("shape", {}).get("r", 0)

        img = UniversalImage(
            image_path=image_path,
            dataset=dataset,
            width=width,
            height=height
        )

        for obj in anns:

            img.objects.append(

                BoundingBox(

                    class_id=int(obj[0]),

                    class_name=classes.get(
                        int(obj[0]),
                        f"Class_{int(obj[0])}"
                    ),

                    xc=obj[1],

                    yc=obj[2],

                    w=obj[3],

                    h=obj[4],

                    confidence=1.0,

                    source=dataset

                )

            )

        return img


# ============================================================
# Clinical Adapter
# ============================================================

# ============================================================
# COCO Adapter
# ============================================================

class CocoAdapter(BaseAdapter):
    """
    Adapter for COCO-format datasets: one JSON file describes every
    image in the dataset, unlike per-image formats (YOLO/Pascal
    VOC). The pipeline still calls `convert()` once per image with
    an `annotation_file` path (matched by FileMatcher, typically the
    same COCO json matched to every image) -- so this adapter parses
    that file once and caches the result, keyed by file path, rather
    than re-parsing the whole (potentially large) COCO file on every
    single image.
    """

    def __init__(self):

        self._cached_path = None
        self._cached_images_by_filename = None
        self._cached_categories = None

    def _load(self, annotation_file):

        annotation_file = str(annotation_file)

        if self._cached_path == annotation_file:
            return self._cached_images_by_filename, self._cached_categories

        with open(annotation_file, "r") as f:
            data = json.load(f)

        images_by_filename, categories = parse_coco_json(data)

        self._cached_path = annotation_file
        self._cached_images_by_filename = images_by_filename
        self._cached_categories = categories

        return images_by_filename, categories

    def convert(
        self,
        annotation_file,
        image_path="",
        dataset="COCO",
    ):

        images_by_filename, categories = self._load(annotation_file)

        filename = Path(image_path).name

        record = images_by_filename.get(filename)

        if record is None:
            # Fall back to matching by stem, in case file extensions
            # differ between the COCO record and the actual file on
            # disk (a real-world quirk seen in some exported datasets).
            stem = Path(image_path).stem
            for candidate_name, candidate_record in images_by_filename.items():
                if Path(candidate_name).stem == stem:
                    record = candidate_record
                    break

        width = record["width"] if record else 0
        height = record["height"] if record else 0

        img = UniversalImage(
            image_path=image_path,
            dataset=dataset,
            width=width,
            height=height,
        )

        if record is None:
            return img

        for annotation in record["annotations"]:

            category_id = annotation.get("category_id")

            class_name = categories.get(category_id, f"Class_{category_id}")

            bbox = annotation.get("bbox", [0, 0, 0, 0])
            x, y, box_w, box_h = bbox

            if width and height:
                xc = (x + box_w / 2) / width
                yc = (y + box_h / 2) / height
                wn = box_w / width
                hn = box_h / height
            else:
                xc = yc = wn = hn = 0.0

            img.objects.append(
                BoundingBox(
                    class_id=int(category_id) if category_id is not None else -1,
                    class_name=class_name,
                    xc=xc,
                    yc=yc,
                    w=wn,
                    h=hn,
                    confidence=1.0,
                    source=dataset,
                )
            )

        return img


# ============================================================
# RBC Morphology Adapter
# ============================================================

class RBCMorphologyAdapter(BaseAdapter):
    """
    Adapter for the RBCMorphology dataset's actual file format:
    per-image point annotations (`x y class`), structurally identical
    to Chula_RBC. Uses `parse_rbc_morphology_txt`, which already
    existed and already matches this dataset's registry class list
    (0: Normal ... 4: Polychromasia) -- but was never wired to any
    adapter before this fix.

    Note: `dataset_registry.py` labels this dataset task="Clinical",
    annotation="CBC Report", which described a different, never-
    finished design (see ClinicalAdapter below). This adapter builds
    the dataset from the file format that actually exists.
    """

    def convert(
        self,
        annotation_file,
        image_path="",
        dataset="RBCMorphology",
        width=640,
        height=480
    ):

        anns = parse_rbc_morphology_txt(annotation_file)

        classes = get_classes(dataset)

        img = UniversalImage(
            image_path=image_path,
            dataset=dataset,
            width=width,
            height=height
        )

        for obj in anns:

            img.objects.append(

                BoundingBox(

                    class_id=int(obj[0]),

                    class_name=classes.get(
                        int(obj[0]),
                        f"Class_{int(obj[0])}"
                    ),

                    xc=obj[1],

                    yc=obj[2],

                    w=obj[3],

                    h=obj[4],

                    confidence=1.0,

                    source=dataset

                )

            )

        return img


# ============================================================
# Clinical Adapter
# ============================================================

class ClinicalAdapter(BaseAdapter):
    """
    Adapter for a CBC-report-style clinical dataset: one text file per
    image containing both numeric CBC fields (Hemoglobin, RBC_Count,
    ...) and an RBC morphology findings line. See parse_cbc_report()
    and extract_morphology_findings() in parsers.py for the expected
    format.

    Note: this previously required two separate files (morphology_file,
    cbc_file) as required positional arguments, and called
    extract_morphology_findings()/parse_cbc_report(), neither of which
    was defined anywhere in the codebase -- meaning this adapter could
    never have run, not even in isolation. Rewritten to accept a single
    `annotation_file`, matching the interface pipeline.py actually uses
    for every adapter, and both parser functions are now implemented.

    Not currently wired to any dataset in builder_registry.py --
    RBCMorphology uses RBCMorphologyAdapter instead, since its real
    files are point annotations, not CBC reports. Kept here, fixed and
    working, for a future dataset that actually ships in this format.
    """

    def convert(
        self,
        annotation_file,
        image_path="",
        dataset="Clinical",
    ):

        morphology = extract_morphology_findings(
            annotation_file
        )

        cbc = parse_cbc_report(
            annotation_file
        )

        return ClinicalRecord(

            image_path=image_path,

            dataset=dataset,

            morphology=morphology,

            cbc=cbc

        )
