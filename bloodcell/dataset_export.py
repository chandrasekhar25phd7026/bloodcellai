"""
==============================================================
BloodCellAI Dataset Export Module (Phase 5, part 1)
==============================================================

File:
    dataset_export.py

Description
-----------
Export an already-built, harmonized UniversalDataset into the file
formats real training tools expect:

    - export_yolo()               -> YOLOv5/v8/ultralytics-style
                                      directory + data.yaml
    - export_classification_folders() -> folder-per-class, per split

Both reuse UniversalDataset.train_set()/val_set()/test_set() (Phase 1)
for splitting, so the exported data respects whatever split was
assigned via assign_splits() -- including per-source-dataset
stratification for multi-dataset studies.

This module has no training-framework dependency (no torch/ultralytics
required) -- it only writes files in the shape those tools expect, so
it works everywhere bloodcell itself works.

Author:
    BloodCellAI Project
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


# =============================================================================
# YOLO Export (Detection)
# =============================================================================

def export_yolo(
    dataset,
    output_dir,
    class_names: Optional[list] = None,
    copy_images: bool = True,
    preprocessing_config=None,
) -> Path:
    """
    Export a detection-task UniversalDataset into a standard
    YOLOv5/v8/ultralytics-style directory:

        output_dir/
            data.yaml
            train/images/*.jpg  train/labels/*.txt
            val/images/*.jpg    val/labels/*.txt
            test/images/*.jpg   test/labels/*.txt

    Images without an assigned split (image.split falsy) are treated
    as "train" -- matches UniversalImage's own default.

    Parameters
    ----------
    dataset : UniversalDataset
        Must already have real bounding-box objects (BoundingBox with
        class_id/xc/yc/w/h) -- classification-only datasets (whole-
        image label as a full-frame box) can still be exported this
        way if desired, but export_classification_folders() is the
        more natural fit for those.

    output_dir : str or Path

    class_names : list[str], optional
        Ordered class names for data.yaml. If not given, derived from
        the dataset's own class_counts, ordered by the numeric
        class_id observed on each BoundingBox (falls back to
        alphabetical if class_id isn't consistently available).

    copy_images : bool
        If True (default), copies image files into the export
        directory. If False, writes label files only and leaves
        images where they are (data.yaml still points at the copied
        layout, so this is mainly useful for a dry run / label-only
        re-export). Ignored if preprocessing_config is given.

    preprocessing_config : preprocessing.PreprocessingConfig, optional
        If given, every image is run through the full preprocessing
        pipeline (resize/CLAHE/denoise/sharpen/color-balance/normalize)
        at export time, and the RESULT is written to the training
        directory -- not a raw copy of the original file. This is
        what actually makes "does preprocessing improve training
        outcomes" a testable question: without this, export_yolo()
        always wrote a raw copy regardless of whether
        UniversalBuilder.enable_preprocessing() was used during the
        build step, so a preprocessing on/off ablation would have
        compared byte-identical training data both times.

    Returns
    -------
    Path
        The output_dir, for convenience chaining.
    """

    output_dir = Path(output_dir)

    splits = {
        "train": dataset.train_set(),
        "val": dataset.val_set(),
        "test": dataset.test_set(),
    }

    # Images with no split assigned at all default to "train",
    # matching UniversalImage.split's own default -- make sure they
    # aren't silently dropped from the export.
    assigned_ids = {
        img.image_id
        for split_dataset in splits.values()
        for img in split_dataset
    }
    unassigned = dataset.find(lambda img: img.image_id not in assigned_ids)

    if unassigned:
        for img in unassigned:
            splits["train"].add(img)

    id_to_name = _resolve_class_names(dataset, class_names)

    # YOLO's data.yaml/label-file format requires class ids to be a
    # contiguous 0..N-1 range matching `nc`. BloodCellAI's own internal
    # class ids are not guaranteed to be contiguous -- e.g. the real
    # Malaria_BBoxes dataset uses ids {0: RBC, 1: WBC, 3: Parasite}
    # (no class 2, since this dataset has no platelets), consistent
    # with the shared CLASS_MAP used across all datasets, but that "3"
    # exceeds nc=3's valid range of 0-2 for YOLO specifically. This
    # remaps to a contiguous space for the export only; BloodCellAI's
    # own internal ids are untouched.
    original_ids_sorted = sorted(id_to_name.keys())
    original_to_contiguous = {
        original_id: contiguous_id
        for contiguous_id, original_id in enumerate(original_ids_sorted)
    }
    contiguous_id_to_name = {
        contiguous_id: id_to_name[original_id]
        for original_id, contiguous_id in original_to_contiguous.items()
    }

    preprocessing_manager = None
    if preprocessing_config is not None:
        from preprocessing.preprocessing_manager import PreprocessingManager
        preprocessing_manager = PreprocessingManager(preprocessing_config)

    for split_name, split_dataset in splits.items():

        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image in split_dataset:

            _write_yolo_label(image, labels_dir, original_to_contiguous)

            if preprocessing_manager is not None:
                _write_processed_image(image, images_dir, preprocessing_manager)
            elif copy_images and Path(image.image_path).exists():
                dest = images_dir / Path(image.image_path).name
                shutil.copy2(image.image_path, dest)

    _write_data_yaml(output_dir, contiguous_id_to_name)

    return output_dir


def _write_processed_image(image, images_dir: Path, preprocessing_manager) -> None:
    """
    Run the full preprocessing pipeline on one image and write the
    actual resulting pixels to images_dir, instead of copying the
    original file -- this is what makes exported "preprocessing on"
    data genuinely different from "preprocessing off" data.

    Falls back to a raw copy (with a printed warning) if preprocessing
    fails for this specific image, so one bad file doesn't halt an
    entire export.
    """

    import cv2
    import numpy as np

    src_path = Path(image.image_path)
    dest_path = images_dir / src_path.name

    try:
        result = preprocessing_manager.preprocess_image(str(src_path))
        processed = result.processed_image

        # Pipeline output may be a normalized float array (e.g. 0-1
        # range from NormalizeTransform) -- convert back to a real,
        # savable 8-bit image regardless of its current range.
        if processed.dtype != np.uint8:
            processed = processed.astype(np.float32)
            max_value = processed.max() if processed.size else 1.0
            if max_value <= 1.0 + 1e-6:
                processed = processed * 255.0
            processed = np.clip(processed, 0, 255).astype(np.uint8)

        cv2.imwrite(str(dest_path), processed)

    except Exception as exc:
        print(f"WARNING: preprocessing failed for {src_path.name} ({exc}); using raw copy.")
        if src_path.exists():
            shutil.copy2(src_path, dest_path)


def _write_yolo_label(image, labels_dir: Path, original_to_contiguous: dict) -> None:

    label_path = labels_dir / f"{Path(image.image_path).stem}.txt"

    lines = []

    for obj in getattr(image, "objects", []):
        contiguous_id = original_to_contiguous.get(obj.class_id, obj.class_id)
        lines.append(
            f"{contiguous_id} {obj.xc:.6f} {obj.yc:.6f} {obj.w:.6f} {obj.h:.6f}"
        )

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _resolve_class_names(dataset, class_names: Optional[list]) -> dict:

    if class_names is not None:
        return {i: name for i, name in enumerate(class_names)}

    id_to_name = {}

    for image in dataset.images:
        for obj in getattr(image, "objects", []):
            if obj.class_id not in id_to_name:
                id_to_name[obj.class_id] = obj.class_name

    if not id_to_name:
        # No objects anywhere (e.g. empty dataset) -- fall back to
        # whatever class names are known at all, in alphabetical order.
        return {i: name for i, name in enumerate(sorted(dataset.class_counts.keys()))}

    return dict(sorted(id_to_name.items()))


def _write_data_yaml(output_dir: Path, id_to_name: dict) -> None:

    names_list = [id_to_name[i] for i in sorted(id_to_name.keys())]

    lines = [
        f"train: {output_dir / 'train' / 'images'}",
        f"val: {output_dir / 'val' / 'images'}",
        f"test: {output_dir / 'test' / 'images'}",
        f"nc: {len(names_list)}",
        f"names: {names_list!r}",
    ]

    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n")


# =============================================================================
# Classification Folder Export
# =============================================================================

def export_classification_folders(
    dataset,
    output_dir,
    copy_images: bool = True,
) -> Path:
    """
    Export a classification-task UniversalDataset (built via
    ClassificationAdapter's whole-image-as-full-frame-box convention)
    into a standard folder-per-class layout, per split:

        output_dir/
            train/<ClassName>/*.jpg
            val/<ClassName>/*.jpg
            test/<ClassName>/*.jpg

    This is the layout torchvision.datasets.ImageFolder (and most
    classification training scripts) expect directly.
    """

    output_dir = Path(output_dir)

    splits = {
        "train": dataset.train_set(),
        "val": dataset.val_set(),
        "test": dataset.test_set(),
    }

    assigned_ids = {
        img.image_id
        for split_dataset in splits.values()
        for img in split_dataset
    }
    unassigned = dataset.find(lambda img: img.image_id not in assigned_ids)
    if unassigned:
        for img in unassigned:
            splits["train"].add(img)

    for split_name, split_dataset in splits.items():

        for image in split_dataset:

            objects = getattr(image, "objects", [])
            class_name = objects[0].class_name if objects else "unlabeled"

            class_dir = output_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            if copy_images and Path(image.image_path).exists():
                dest = class_dir / Path(image.image_path).name
                shutil.copy2(image.image_path, dest)

    return output_dir
