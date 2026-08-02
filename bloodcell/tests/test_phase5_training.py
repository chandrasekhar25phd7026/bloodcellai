"""
Tests for Phase 5: patch_features, classifier, explainability
(permutation importance path), and dataset_export.
"""

import numpy as np
import cv2


def _make_synthetic_bccd_style_dataset(tmp_path, n_per_class=20):
    """
    Build a small synthetic detection dataset with three visually
    distinct "classes" (different shapes/colors), so a classifier
    trained on it should do meaningfully better than chance --
    verifies the whole feature-extraction + training wiring without
    needing the full real BCCD set for a fast unit test.
    """

    from bloodcell.universal_object import UniversalImage, BoundingBox
    from bloodcell.universal_dataset import UniversalDataset

    rng = np.random.default_rng(0)

    dataset = UniversalDataset()

    colors = {"ClassA": (200, 50, 50), "ClassB": (50, 200, 50), "ClassC": (50, 50, 200)}

    for i in range(n_per_class * 3):

        class_name = list(colors.keys())[i % 3]

        img = np.full((128, 128, 3), 255, dtype=np.uint8)
        color = colors[class_name]
        cv2.circle(img, (64, 64), 30, color, -1)
        noise = rng.normal(0, 5, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(path), img)

        ui = UniversalImage(image_path=str(path), dataset="synthetic", width=128, height=128)
        ui.objects.append(
            BoundingBox(class_id=i % 3, class_name=class_name, xc=0.5, yc=0.5, w=0.5, h=0.5)
        )
        dataset.add(ui)

    return dataset


def test_crop_patch_produces_valid_region():
    from training.patch_features import crop_patch
    from bloodcell.universal_object import BoundingBox

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    obj = BoundingBox(class_id=0, class_name="X", xc=0.5, yc=0.5, w=0.2, h=0.2)

    patch = crop_patch(image, obj)

    assert patch is not None
    assert patch.shape[0] > 0
    assert patch.shape[1] > 0


def test_extract_features_returns_fixed_length_vector():
    from training.patch_features import extract_features, FEATURE_NAMES

    patch = np.random.default_rng(0).integers(0, 255, (50, 50, 3), dtype=np.uint8)
    features = extract_features(patch)

    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_build_feature_dataset_from_real_structure(tmp_path):
    from training.patch_features import build_feature_dataset

    dataset = _make_synthetic_bccd_style_dataset(tmp_path, n_per_class=10)

    X, y, class_names = build_feature_dataset(dataset)

    assert X.shape[0] == 30
    assert set(class_names) == {"ClassA", "ClassB", "ClassC"}


def test_max_objects_per_class_caps_correctly(tmp_path):
    from training.patch_features import build_feature_dataset

    dataset = _make_synthetic_bccd_style_dataset(tmp_path, n_per_class=20)

    X, y, class_names = build_feature_dataset(dataset, max_objects_per_class=5)

    assert X.shape[0] == 15  # 5 per class * 3 classes


def test_classifier_trains_and_evaluates_above_chance(tmp_path):
    from training.patch_features import build_feature_dataset
    from training.classifier import train_and_evaluate

    dataset = _make_synthetic_bccd_style_dataset(tmp_path, n_per_class=30)
    X, y, class_names = build_feature_dataset(dataset)

    model, report = train_and_evaluate(X, y, class_names, test_size=0.3)

    # Three visually distinct solid colors should be trivially
    # separable -- well above chance (1/3).
    assert report.accuracy > 0.7
    assert len(report.confusion_matrix) == 3


def test_permutation_importance_runs_and_ranks_features(tmp_path):
    from training.patch_features import build_feature_dataset, FEATURE_NAMES
    from training.classifier import train_and_evaluate
    from training.explainability import permutation_importance_report

    dataset = _make_synthetic_bccd_style_dataset(tmp_path, n_per_class=30)
    X, y, class_names = build_feature_dataset(dataset)

    model, report = train_and_evaluate(X, y, class_names)

    importance = permutation_importance_report(model, X, y, FEATURE_NAMES, n_repeats=3)

    assert len(importance.ranked()) == len(FEATURE_NAMES)
    # Color-based features should matter for solid-color-circle classes.
    top_feature_names = {name for name, _, _ in importance.ranked()[:5]}
    assert any("mean" in name or "hue" in name for name in top_feature_names)


def test_export_yolo_produces_valid_structure(tmp_path):
    from bloodcell.universal_object import UniversalImage, BoundingBox
    from bloodcell.universal_dataset import UniversalDataset
    from bloodcell.dataset_export import export_yolo

    dataset = UniversalDataset()

    for i in range(6):
        img_path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(img_path), np.zeros((64, 64, 3), dtype=np.uint8))

        ui = UniversalImage(image_path=str(img_path), dataset="X", width=64, height=64)
        ui.objects.append(
            BoundingBox(class_id=0, class_name="RBC", xc=0.5, yc=0.5, w=0.1, h=0.1)
        )
        dataset.add(ui)

    dataset.assign_splits(train=0.7, val=0.15, test=0.15, seed=0)

    output_dir = tmp_path / "yolo_export"
    export_yolo(dataset, output_dir)

    assert (output_dir / "data.yaml").exists()
    assert (output_dir / "train" / "images").is_dir()
    assert (output_dir / "train" / "labels").is_dir()

    total_labels = sum(
        len(list((output_dir / split / "labels").glob("*.txt")))
        for split in ("train", "val", "test")
    )
    assert total_labels == 6

    data_yaml = (output_dir / "data.yaml").read_text()
    assert "RBC" in data_yaml


def test_export_yolo_remaps_non_contiguous_class_ids(tmp_path):
    """
    Regression test: BloodCellAI's internal class ids (shared via
    CLASS_MAP across datasets) are not guaranteed to be contiguous --
    confirmed on real data: Malaria_BBoxes uses ids {0: RBC, 1: WBC,
    3: Parasite} (no class 2, since this dataset has no platelets).
    YOLO's data.yaml/label format requires contiguous 0..N-1 ids
    matching nc -- writing the raw id "3" with nc=3 caused Ultralytics
    to reject every Parasite-labeled object as corrupt on real data.
    export_yolo() must remap to a contiguous space for the export.
    """
    from bloodcell.universal_object import UniversalImage, BoundingBox
    from bloodcell.universal_dataset import UniversalDataset
    from bloodcell.dataset_export import export_yolo

    dataset = UniversalDataset()

    specs = [(0, "RBC"), (1, "WBC"), (3, "Parasite")]
    for i, (cid, name) in enumerate(specs):
        path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))
        img = UniversalImage(image_path=str(path), dataset="X", width=64, height=64)
        img.objects.append(
            BoundingBox(class_id=cid, class_name=name, xc=0.5, yc=0.5, w=0.1, h=0.1)
        )
        dataset.add(img)

    dataset.assign_splits(train=0.34, val=0.33, test=0.33, seed=0, stratify_by=None)

    output_dir = tmp_path / "export"
    export_yolo(dataset, output_dir)

    data_yaml = (output_dir / "data.yaml").read_text()
    assert "nc: 3" in data_yaml
    assert "Parasite" in data_yaml

    # Every class id actually written to a label file must be < nc (3),
    # i.e. in {0, 1, 2} -- never the original raw id 3.
    max_class_id_found = -1
    for split in ("train", "val", "test"):
        labels_dir = output_dir / split / "labels"
        for label_file in labels_dir.glob("*.txt"):
            content = label_file.read_text().strip()
            if content:
                class_id = int(content.split()[0])
                max_class_id_found = max(max_class_id_found, class_id)

    assert max_class_id_found < 3, (
        f"Found class id {max_class_id_found}, which would exceed nc=3 "
        "and be rejected by YOLO -- the non-contiguous id (3) was not "
        "remapped correctly."
    )
