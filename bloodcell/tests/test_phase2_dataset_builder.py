"""
Tests for Phase 2 additions:
- annotation_intelligence.detect_dataset_format() (whole-folder
  format auto-detection: YOLO, COCO, Pascal VOC, CSV classification,
  folder-per-class classification, Roboflow-style exports)
- dataset_builder.UniversalDatasetBuilder.auto_register() / prepare()
- adapters.CocoAdapter / parsers.parse_coco_json
"""

import json
import pathlib

from PIL import Image

from bloodcell.annotation_intelligence import detect_dataset_format
from bloodcell.dataset_builder import UniversalDatasetBuilder
from bloodcell.adapters import CocoAdapter
from bloodcell.parsers import parse_coco_json


# ---------------------------------------------------------------------------
# detect_dataset_format
# ---------------------------------------------------------------------------

def test_detects_yolo_folder(tmp_path: pathlib.Path):
    for i in range(3):
        Image.new("RGB", (640, 480)).save(tmp_path / f"img{i}.jpg")
        (tmp_path / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    result = detect_dataset_format(tmp_path)

    assert result["task"] == "Detection"
    assert result["annotation"] == "YOLO"
    assert result["is_roboflow_export"] is False


def test_detects_coco_folder(tmp_path: pathlib.Path):
    for i in range(2):
        Image.new("RGB", (640, 480)).save(tmp_path / f"img{i}.jpg")

    coco = {
        "images": [{"id": 0, "file_name": "img0.jpg", "width": 640, "height": 480}],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20]}],
        "categories": [{"id": 1, "name": "RBC"}],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco))

    result = detect_dataset_format(tmp_path)

    assert result["task"] == "Detection"
    assert result["annotation"] == "COCO"


def test_detects_folder_per_class(tmp_path: pathlib.Path):
    for cls in ("Neutrophil", "Lymphocyte"):
        (tmp_path / cls).mkdir()
        Image.new("RGB", (200, 200)).save(tmp_path / cls / "a.jpg")

    result = detect_dataset_format(tmp_path)

    assert result["task"] == "Classification"
    assert result["annotation"] == "None"
    assert set(result["classes"].values()) == {"Neutrophil", "Lymphocyte"}


def test_detects_csv_classification(tmp_path: pathlib.Path):
    Image.new("RGB", (200, 200)).save(tmp_path / "0.bmp")
    Image.new("RGB", (200, 200)).save(tmp_path / "1.bmp")
    (tmp_path / "labels.csv").write_text("image ID,class label\n0,1\n1,2\n")

    result = detect_dataset_format(tmp_path)

    assert result["task"] == "Classification"
    assert result["annotation"] == "CSV"


def test_detects_roboflow_export_with_manifest(tmp_path: pathlib.Path):
    images_dir = tmp_path / "train" / "images"
    labels_dir = tmp_path / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for i in range(2):
        Image.new("RGB", (640, 640)).save(images_dir / f"img{i}.jpg")
        (labels_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    (tmp_path / "data.yaml").write_text(
        "train: ../train/images\nnc: 2\nnames: ['RBC', 'WBC']\n"
    )

    result = detect_dataset_format(tmp_path)

    assert result["is_roboflow_export"] is True
    assert result["annotation"] == "YOLO"
    assert result["classes"] == {0: "RBC", 1: "WBC"}


def test_detects_unknown_for_empty_folder(tmp_path: pathlib.Path):
    result = detect_dataset_format(tmp_path)

    assert result["annotation"] == "Unknown"
    assert result["evidence"]  # non-empty explanation


# ---------------------------------------------------------------------------
# UniversalDatasetBuilder auto_register / prepare
# ---------------------------------------------------------------------------

def test_auto_register_and_prepare(tmp_path: pathlib.Path):
    for i in range(3):
        Image.new("RGB", (640, 480)).save(tmp_path / f"img{i}.jpg")
        (tmp_path / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    builder = UniversalDatasetBuilder()
    ds = builder.auto_register(tmp_path, dataset_name="my_dataset")

    assert ds.task == "Detection"
    assert ds.annotation == "YOLO"
    assert builder.get_dataset("my_dataset") is ds

    prepared = builder.prepare("my_dataset")

    assert prepared.image_count == 3
    assert prepared.annotation_count == 3
    assert prepared.status == "READY"


def test_prepare_all_continues_past_one_bad_dataset(tmp_path: pathlib.Path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    Image.new("RGB", (640, 480)).save(good_dir / "img0.jpg")
    (good_dir / "img0.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    builder = UniversalDatasetBuilder()
    builder.auto_register(good_dir, dataset_name="good")
    builder.auto_register(empty_dir, dataset_name="empty")

    results = builder.prepare_all()

    assert results["good"].status == "READY"
    # empty folder -> Unknown task/annotation, but must not crash
    # prepare_all() or prevent "good" from being prepared.
    assert "empty" in results


def test_get_dataset_returns_none_for_unregistered():
    builder = UniversalDatasetBuilder()
    assert builder.get_dataset("nonexistent") is None


# ---------------------------------------------------------------------------
# COCO parser / adapter
# ---------------------------------------------------------------------------

def test_parse_coco_json_indexes_by_filename():
    data = {
        "images": [{"id": 5, "file_name": "x.jpg", "width": 100, "height": 200}],
        "annotations": [
            {"id": 0, "image_id": 5, "category_id": 9, "bbox": [10, 20, 30, 40]}
        ],
        "categories": [{"id": 9, "name": "Platelet"}],
    }

    images_by_filename, categories = parse_coco_json(data)

    assert "x.jpg" in images_by_filename
    assert images_by_filename["x.jpg"]["width"] == 100
    assert len(images_by_filename["x.jpg"]["annotations"]) == 1
    assert categories[9] == "Platelet"


def test_coco_adapter_converts_bbox_correctly(tmp_path: pathlib.Path):
    data = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 200, "height": 100}],
        "annotations": [
            # absolute pixel bbox: x=50, y=25, w=20, h=10
            # -> center = (60, 30), normalized: xc=0.3, yc=0.3, w=0.1, h=0.1
            {"id": 0, "image_id": 1, "category_id": 3, "bbox": [50, 25, 20, 10]}
        ],
        "categories": [{"id": 3, "name": "WBC"}],
    }

    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(data))

    adapter = CocoAdapter()
    image = adapter.convert(
        annotation_file=ann_file,
        image_path=str(tmp_path / "sample.jpg"),
        dataset="TestCOCO",
    )

    assert image.width == 200
    assert image.height == 100
    assert len(image.objects) == 1

    obj = image.objects[0]
    assert obj.class_name == "WBC"
    assert abs(obj.xc - 0.3) < 1e-6
    assert abs(obj.yc - 0.3) < 1e-6
    assert abs(obj.w - 0.1) < 1e-6
    assert abs(obj.h - 0.1) < 1e-6


def test_coco_adapter_caches_across_multiple_images(tmp_path: pathlib.Path):
    """
    A COCO file describes the WHOLE dataset -- converting several
    images that share one annotation file should parse that file
    only once, not once per image.
    """

    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "annotations": [],
        "categories": [],
    }

    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(data))

    adapter = CocoAdapter()

    adapter.convert(annotation_file=ann_file, image_path=str(tmp_path / "a.jpg"), dataset="X")
    cached_data_id_after_first = id(adapter._cached_images_by_filename)

    adapter.convert(annotation_file=ann_file, image_path=str(tmp_path / "b.jpg"), dataset="X")
    cached_data_id_after_second = id(adapter._cached_images_by_filename)

    # Same cached dict object reused -- confirms no re-parse happened.
    assert cached_data_id_after_first == cached_data_id_after_second


# ---------------------------------------------------------------------------
# Regression tests: whole-dataset-annotation-format fix (COCO/Malaria JSON)
# ---------------------------------------------------------------------------

def test_build_from_info_coco_whole_dataset_file(tmp_path: pathlib.Path):
    """
    Regression test: FileMatcher matches annotations to images by
    filename stem, which cannot find a single shared COCO file (whose
    stem is "annotations", not any image's stem) except by
    coincidence. build_from_info() must locate the shared file
    directly rather than relying on FileMatcher for this format --
    confirmed broken (0 images built) before this fix.
    """

    from bloodcell.dataset_builder import UniversalDatasetBuilder
    from bloodcell.universal_builder import UniversalBuilder

    for i in range(3):
        Image.new("RGB", (640, 480)).save(tmp_path / f"img{i}.jpg")

    coco = {
        "images": [
            {"id": i, "file_name": f"img{i}.jpg", "width": 640, "height": 480}
            for i in range(3)
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 10, 20, 20]}
        ],
        "categories": [{"id": 1, "name": "RBC"}],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco))

    db = UniversalDatasetBuilder()
    info = db.auto_register(tmp_path, dataset_name="coco_whole_dataset_test")
    db.prepare("coco_whole_dataset_test")

    ub = UniversalBuilder(tmp_path)
    dataset, built = ub.build_from_info(info)

    assert built == 3
    assert dataset.class_counts["RBC"] == 1


def test_malaria_adapter_handles_real_list_shaped_file(tmp_path: pathlib.Path):
    """
    Regression test: the real public Malaria Bounding Boxes dataset
    is a JSON LIST of many per-image records in one shared file, not
    one record per file -- confirmed from the real dataset structure.
    An earlier version of this adapter assumed a single record
    (`json_record["image"]`), which crashes immediately on the real
    list-shaped file.
    """

    from bloodcell.adapters import MalariaAdapter

    data = [
        {
            "image": {"pathname": "/images/a.png", "shape": {"r": 100, "c": 200}},
            "objects": [
                {"category": "red blood cell", "bounding_box": {
                    "minimum": {"r": 10, "c": 10}, "maximum": {"r": 20, "c": 20}
                }}
            ],
        },
        {
            "image": {"pathname": "/images/b.png", "shape": {"r": 100, "c": 200}},
            "objects": [],
        },
    ]

    ann_file = tmp_path / "training.json"
    ann_file.write_text(json.dumps(data))

    adapter = MalariaAdapter()

    image_a = adapter.convert(
        annotation_file=ann_file, image_path=str(tmp_path / "a.png"), dataset="Malaria Bounding Boxes"
    )
    image_b = adapter.convert(
        annotation_file=ann_file, image_path=str(tmp_path / "b.png"), dataset="Malaria Bounding Boxes"
    )

    assert image_a.width == 200
    assert image_a.height == 100
    assert len(image_a.objects) == 1
    assert len(image_b.objects) == 0


def test_voc_class_extraction_matches_class_map_not_appearance_order(tmp_path: pathlib.Path):
    """
    Regression test: class ids extracted from VOC <name> tags must
    match the global CLASS_MAP that parse_pascal_xml() actually uses
    (RBC=0, WBC=1, Platelet=2), NOT first-appearance order in the
    sampled files. Assigning by appearance order silently swapped
    WBC and Platelet labels on real BCCD data before this fix, since
    "Platelets" happened to appear before "WBC" in the sample.
    """

    from bloodcell.annotation_intelligence import _extract_voc_class_names

    xml_platelets_first = tmp_path / "a.xml"
    xml_platelets_first.write_text(
        "<annotation><object><name>Platelets</name></object></annotation>"
    )

    xml_wbc_second = tmp_path / "b.xml"
    xml_wbc_second.write_text(
        "<annotation><object><name>WBC</name></object></annotation>"
    )

    classes = _extract_voc_class_names([xml_platelets_first, xml_wbc_second])

    assert classes[1] == "WBC"        # CLASS_MAP["WBC"] == 1
    assert classes[2] == "Platelets"  # CLASS_MAP["Platelets"] == 2


def test_whole_dataset_json_merges_multiple_files(tmp_path):
    """
    Regression test: some real datasets (confirmed: the real Malaria
    Bounding Boxes dataset) split whole-dataset annotations across
    multiple files (training.json + test.json) covering different,
    non-overlapping images. Taking only the first matching file
    silently discarded every record in the others -- confirmed on
    real data to drop 1,208 real annotated records. Fixed by merging
    all matching files.
    """
    import json
    from bloodcell.universal_builder import _find_whole_dataset_annotation_file

    training_data = [
        {"image": {"pathname": f"/img_train_{i}.png", "shape": {"r": 100, "c": 100}},
         "objects": [{"category": "red blood cell", "bounding_box": {
             "minimum": {"r": 10, "c": 10}, "maximum": {"r": 20, "c": 20}}}]}
        for i in range(5)
    ]
    test_data = [
        {"image": {"pathname": f"/img_test_{i}.png", "shape": {"r": 100, "c": 100}},
         "objects": [{"category": "leukocyte", "bounding_box": {
             "minimum": {"r": 30, "c": 30}, "maximum": {"r": 40, "c": 40}}}]}
        for i in range(2)
    ]

    (tmp_path / "training.json").write_text(json.dumps(training_data))
    (tmp_path / "test.json").write_text(json.dumps(test_data))

    result_path = _find_whole_dataset_annotation_file(tmp_path, "Malaria JSON")

    with open(result_path) as f:
        merged = json.load(f)

    # Before the fix, only test.json's 2 records would survive (it sorts
    # alphabetically before training.json) -- all 7 must be present now.
    assert len(merged) == 7

    pathnames = {r["image"]["pathname"] for r in merged}
    assert pathnames == {f"/img_train_{i}.png" for i in range(5)} | {f"/img_test_{i}.png" for i in range(2)}
