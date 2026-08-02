"""
==============================================================
BloodCellAI Training — Patch Extraction & Feature Engineering
==============================================================

File:
    patch_features.py

Description
-----------
Turns a detection-task UniversalDataset (bounding boxes) into a
classification-ready feature matrix, by cropping each annotated
object out of its source image and computing a compact set of
engineered visual features per patch.

Why engineered features, not CNN features
------------------------------------------
This project's execution environment has no GPU and no PyTorch/
ultralytics installed (confirmed: no network path to a usable torch
wheel, and insufficient disk space for one). Rather than write CNN
training code that can't actually be run or verified here, this
module implements a genuinely working, classical-ML feature
pipeline using only numpy/opencv/scikit-learn, all of which ARE
installed -- so the training pipeline in this project is real code
that has actually been run against real data, not a plausible-looking
stub. See training/deep_learning.py for the CNN/Grad-CAM code,
written for the GPU environment (Kaggle/Colab) where it can actually
run, clearly marked as such.

Author:
    BloodCellAI Project
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def crop_patch(image_bgr: np.ndarray, obj, padding: float = 0.1) -> "np.ndarray | None":
    """
    Crop one BoundingBox's region out of a full image.

    Parameters
    ----------
    image_bgr : np.ndarray
        Full source image (as loaded by cv2.imread).

    obj : BoundingBox
        Normalized xc/yc/w/h (0-1), matching how BloodCellAI stores
        every object regardless of source annotation format.

    padding : float
        Extra fractional margin added around the box (helps context
        for classification), clamped to stay inside the image.

    Returns
    -------
    np.ndarray or None
        None if the resulting crop would be empty/invalid.
    """

    height, width = image_bgr.shape[:2]

    box_w = obj.w * (1 + padding)
    box_h = obj.h * (1 + padding)

    x1 = int(max(0, (obj.xc - box_w / 2) * width))
    y1 = int(max(0, (obj.yc - box_h / 2) * height))
    x2 = int(min(width, (obj.xc + box_w / 2) * width))
    y2 = int(min(height, (obj.yc + box_h / 2) * height))

    if x2 <= x1 or y2 <= y1:
        return None

    return image_bgr[y1:y2, x1:x2]


def extract_features(patch_bgr: np.ndarray) -> np.ndarray:
    """
    Compute a compact, fixed-length engineered feature vector for one
    image patch.

    Features (18 total)
    --------------------
    - Per-channel (B, G, R) mean and std                  (6)
    - Grayscale mean, std, entropy                          (3)
    - Color histogram, 8 bins, hue channel (HSV)            (8)
    - Edge density (fraction of Canny edge pixels)          (1)

    All are standard, interpretable, classical computer-vision
    features -- deliberately simple and fast, appropriate for the
    "does the wiring work end-to-end" scope of this pipeline; a real
    research deployment would swap this for CNN embeddings once run
    in a GPU environment (see training/deep_learning.py).
    """

    patch = cv2.resize(patch_bgr, (64, 64))

    features = []

    for channel in range(3):
        chan = patch[:, :, channel].astype(np.float32)
        features.append(float(chan.mean()))
        features.append(float(chan.std()))

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    features.append(float(gray.mean()))
    features.append(float(gray.std()))

    histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
    probabilities = histogram / (histogram.sum() + 1e-8)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    features.append(entropy)

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue_histogram, _ = np.histogram(hsv[:, :, 0], bins=8, range=(0, 180))
    hue_histogram = hue_histogram / (hue_histogram.sum() + 1e-8)
    features.extend(hue_histogram.tolist())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    features.append(edge_density)

    return np.array(features, dtype=np.float32)


FEATURE_NAMES = (
    ["B_mean", "B_std", "G_mean", "G_std", "R_mean", "R_std"]
    + ["gray_mean", "gray_std", "gray_entropy"]
    + [f"hue_hist_{i}" for i in range(8)]
    + ["edge_density"]
)


def build_feature_dataset(
    dataset,
    max_objects_per_class: "int | None" = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Crop every object in a detection-task UniversalDataset and build
    a (X, y) feature matrix ready for sklearn.

    Parameters
    ----------
    dataset : bloodcell.universal_dataset.UniversalDataset
        Should already be quality-gated (see
        bloodcell.quality_gate.DatasetQualityGate) so features aren't
        extracted from corrupted images.

    max_objects_per_class : int, optional
        Cap the number of patches per class (useful for quick runs /
        balancing a heavily-imbalanced dataset like BCCD, where RBC
        vastly outnumbers WBC/Platelet).

    Returns
    -------
    tuple
        (X, y, class_names) -- X is (N, 18) float32, y is (N,) int
        class indices, class_names[i] is the name for class index i.
    """

    class_names = sorted(dataset.class_counts.keys())
    class_to_index = {name: i for i, name in enumerate(class_names)}

    per_class_counts = {name: 0 for name in class_names}

    features_list = []
    labels_list = []

    image_cache = {}

    for image in dataset:

        if not getattr(image, "objects", None):
            continue

        image_path = str(image.image_path)

        if image_path not in image_cache:
            image_cache[image_path] = cv2.imread(image_path)

        image_bgr = image_cache[image_path]

        if image_bgr is None:
            continue

        for obj in image.objects:

            class_name = obj.class_name

            if class_name not in class_to_index:
                continue

            if (
                max_objects_per_class is not None
                and per_class_counts[class_name] >= max_objects_per_class
            ):
                continue

            patch = crop_patch(image_bgr, obj)

            if patch is None or patch.size == 0:
                continue

            features_list.append(extract_features(patch))
            labels_list.append(class_to_index[class_name])
            per_class_counts[class_name] += 1

    logger.info("Built feature dataset: %s", per_class_counts)

    if not features_list:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            class_names,
        )

    return (
        np.stack(features_list),
        np.array(labels_list, dtype=np.int64),
        class_names,
    )
