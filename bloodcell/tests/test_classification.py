"""
Tests for classification-task dataset building:
- ClassificationAdapter (whole-image label as a full-image BoundingBox)
- UniversalBuilder.build_classification_dataset() for both the
  folder-per-class ("None") and CSV-labeled ("CSV") conventions used
  by the 8 classification datasets in dataset_registry.py.
"""

import pathlib

from PIL import Image

from bloodcell.adapters import ClassificationAdapter
from bloodcell.universal_builder import UniversalBuilder
import bloodcell.dataset_registry as dataset_registry


def test_classification_adapter_produces_full_image_box():
    adapter = ClassificationAdapter()

    # Register a throwaway dataset for this test.
    dataset_registry.DATASET_REGISTRY["_TestClsAdapter"] = {
        "task": "Classification",
        "annotation": "None",
        "classes": {0: "Normal", 1: "Abnormal"},
    }

    image = adapter.convert(
        image_path="/tmp/x.jpg",
        class_id=1,
        dataset="_TestClsAdapter",
        width=100,
        height=100,
    )

    assert len(image.objects) == 1
    obj = image.objects[0]
    assert obj.class_name == "Abnormal"
    assert (obj.xc, obj.yc, obj.w, obj.h) == (0.5, 0.5, 1.0, 1.0)


def test_folder_per_class_build(tmp_path: pathlib.Path):
    dataset_registry.DATASET_REGISTRY["_TestFolderCls"] = {
        "task": "Classification",
        "annotation": "None",
        "classes": {0: "Neutrophil", 1: "Lymphocyte"},
    }

    root = tmp_path / "datasets" / "_TestFolderCls"
    (root / "Neutrophil").mkdir(parents=True)
    (root / "Lymphocyte").mkdir(parents=True)

    Image.new("RGB", (64, 64)).save(root / "Neutrophil" / "a.jpg")
    Image.new("RGB", (64, 64)).save(root / "Neutrophil" / "b.jpg")
    Image.new("RGB", (64, 64)).save(root / "Lymphocyte" / "a.jpg")

    builder = UniversalBuilder(tmp_path)
    dataset, built = builder.build_classification_dataset("_TestFolderCls")

    assert built == 3
    assert dataset.class_counts["Neutrophil"] == 2
    assert dataset.class_counts["Lymphocyte"] == 1


def test_folder_per_class_unmatched_folder_logged_not_crashed(tmp_path: pathlib.Path):
    dataset_registry.DATASET_REGISTRY["_TestFolderClsBad"] = {
        "task": "Classification",
        "annotation": "None",
        "classes": {0: "Neutrophil"},
    }

    root = tmp_path / "datasets" / "_TestFolderClsBad"
    (root / "SomeUnknownClass").mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(root / "SomeUnknownClass" / "a.jpg")

    builder = UniversalBuilder(tmp_path)
    dataset, built = builder.build_classification_dataset("_TestFolderClsBad")

    assert built == 0
    log = builder.log()
    assert (log["status"] == "FAILED").any()


def test_csv_classification_build(tmp_path: pathlib.Path):
    dataset_registry.DATASET_REGISTRY["_TestCsvCls"] = {
        "task": "Classification",
        "annotation": "CSV",
        "classes": {1: "Blast", 2: "Normal"},
    }

    root = tmp_path / "datasets" / "_TestCsvCls"
    root.mkdir(parents=True)

    Image.new("RGB", (64, 64)).save(root / "001.bmp")
    Image.new("RGB", (64, 64)).save(root / "002.bmp")

    # Deliberately zero-padding-mismatched ids (real quirk found in a
    # real public CSV-labeled WBC dataset): CSV says "1"/"2", files are
    # "001.bmp"/"002.bmp".
    (root / "labels.csv").write_text(
        "image ID,class label\n1,1\n2,2\n"
    )

    builder = UniversalBuilder(tmp_path)
    dataset, built = builder.build_classification_dataset("_TestCsvCls")

    assert built == 2
    assert dataset.class_counts["Blast"] == 1
    assert dataset.class_counts["Normal"] == 1


def test_csv_classification_missing_image_logged_not_crashed(tmp_path: pathlib.Path):
    dataset_registry.DATASET_REGISTRY["_TestCsvClsMissing"] = {
        "task": "Classification",
        "annotation": "CSV",
        "classes": {1: "Blast"},
    }

    root = tmp_path / "datasets" / "_TestCsvClsMissing"
    root.mkdir(parents=True)

    Image.new("RGB", (64, 64)).save(root / "001.bmp")

    (root / "labels.csv").write_text(
        "image ID,class label\n1,1\n999,1\n"
    )

    builder = UniversalBuilder(tmp_path)
    dataset, built = builder.build_classification_dataset("_TestCsvClsMissing")

    assert built == 1
    log = builder.log()
    assert (log["status"] == "FAILED").any()
