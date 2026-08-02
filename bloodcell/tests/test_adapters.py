"""
Unit tests for bloodcell.adapters.

Each test feeds a synthetic annotation file to an adapter and checks
that the resulting UniversalImage/BoundingBox fields are correct. This
is cheap insurance against silent corruption when a real dataset's
annotation format has quirks the adapter doesn't expect.
"""

import json
import pathlib

import pytest
from bloodcell.adapters import YOLOAdapter, ChulaAdapter, MalariaAdapter
from bloodcell.universal_object import UniversalImage


def test_yolo_adapter_parses_bounding_boxes(tmp_path: pathlib.Path):
    ann_file = tmp_path / "sample.txt"
    ann_file.write_text("0 0.5 0.5 0.1 0.2\n1 0.25 0.25 0.05 0.05\n")

    adapter = YOLOAdapter()
    image = adapter.convert(
        annotation_file=ann_file,
        image_path=str(tmp_path / "sample.jpg"),
        dataset="BCCD",
        width=640,
        height=480,
    )

    assert isinstance(image, UniversalImage)
    assert image.dataset == "BCCD"
    assert len(image.objects) == 2

    first = image.objects[0]
    assert first.class_id == 0
    assert first.class_name == "RBC"
    assert first.xc == pytest.approx(0.5)
    assert first.yc == pytest.approx(0.5)


def test_yolo_adapter_skips_blank_lines(tmp_path: pathlib.Path):
    ann_file = tmp_path / "sample.txt"
    ann_file.write_text("0 0.5 0.5 0.1 0.2\n\n\n1 0.1 0.1 0.05 0.05\n")

    adapter = YOLOAdapter()
    image = adapter.convert(
        annotation_file=ann_file,
        image_path=str(tmp_path / "sample.jpg"),
        dataset="BCCD",
    )

    assert len(image.objects) == 2


def test_chula_adapter_parses_point_annotations(tmp_path: pathlib.Path):
    ann_file = tmp_path / "sample.txt"
    # format: x y morphology_class
    ann_file.write_text("100 100 3\n200 150 7\n")

    adapter = ChulaAdapter()
    image = adapter.convert(
        annotation_file=ann_file,
        image_path=str(tmp_path / "sample.jpg"),
        dataset="Chula_RBC",
        width=640,
        height=480,
    )

    assert len(image.objects) == 2
    assert image.objects[0].class_id == 3
    assert image.objects[0].class_name == "Spherocyte"
    assert image.objects[1].class_name == "Tear_Drop_Cell"


def test_malaria_adapter_parses_json_record(tmp_path: pathlib.Path):
    record = {
        "image": {"pathname": "/sample.png", "shape": {"r": 1200, "c": 1600}},
        "objects": [
            {
                "category": "trophozoite",
                "bounding_box": {
                    "minimum": {"r": 100, "c": 100},
                    "maximum": {"r": 200, "c": 220},
                },
            }
        ],
    }

    ann_file = tmp_path / "sample.json"
    ann_file.write_text(json.dumps(record))

    adapter = MalariaAdapter()
    image = adapter.convert(
        annotation_file=ann_file,
        dataset="Malaria Bounding Boxes",
        image_path=str(tmp_path / "sample.png"),
    )

    assert image.width == 1600
    assert image.height == 1200


def test_malaria_adapter_recognizes_real_dataset_vocabulary(tmp_path: pathlib.Path):
    # Real category vocabulary confirmed from a public notebook that used
    # the actual Kaggle/NIH Malaria Bounding Boxes dataset: 'red blood cell',
    # 'leukocyte', 'gametocyte', 'ring', 'schizont', 'trophozoite',
    # 'difficult'. 'leukocyte' was missing from CLASS_MAP (silently
    # dropping every WBC in this dataset) until this pass.
    record = {
        "image": {"pathname": "/sample.png", "shape": {"r": 1200, "c": 1600}},
        "objects": [
            {"category": "red blood cell", "bounding_box": {"minimum": {"r": 0, "c": 0}, "maximum": {"r": 50, "c": 50}}},
            {"category": "leukocyte", "bounding_box": {"minimum": {"r": 60, "c": 60}, "maximum": {"r": 120, "c": 120}}},
            {"category": "difficult", "bounding_box": {"minimum": {"r": 200, "c": 200}, "maximum": {"r": 240, "c": 240}}},
        ],
    }

    ann_file = tmp_path / "sample.json"
    ann_file.write_text(json.dumps(record))

    adapter = MalariaAdapter()
    image = adapter.convert(
        annotation_file=ann_file,
        dataset="Malaria Bounding Boxes",
        image_path=str(tmp_path / "sample.png"),
    )

    class_names = {obj.class_name for obj in image.objects}

    assert "RBC" in class_names
    assert "WBC" in class_names
    # 'difficult' is an annotation-ambiguity flag, not a cell type -- it
    # is intentionally left unmapped (and now logged, not silently
    # dropped), so it should not appear as an object.
    assert len(image.objects) == 2
