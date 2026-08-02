"""
==============================================================
BloodCellAI Classical Baseline Classifier (Phase 5, part 2)
==============================================================

File:
    baseline_classifier.py

Description
-----------
A real, working classification baseline that needs no deep-learning
framework (no torch/ultralytics) -- useful as:

    1. A genuinely trainable "does this dataset separate at all"
       sanity check before investing in a full deep-learning run.
    2. A fallback baseline for environments without GPU access.
    3. A reference implementation you can actually run to completion
       in a CPU-only environment (e.g. this sandbox) end-to-end.

Features: HOG (shape/texture) + color histogram (stain/color
signature), fed into an sklearn RandomForestClassifier.

For blood cell classification, published literature and common
practice show hand-crafted feature + classical ML pipelines are a
legitimate, citable baseline (not just a toy) -- this is a fair
comparison point for a deep-learning paper's ablation, not merely
scaffolding.

Author:
    BloodCellAI Project
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# =============================================================================
# Feature Extraction
# =============================================================================

def _hog_features(gray_image: np.ndarray) -> np.ndarray:
    """
    HOG features via OpenCV's HOGDescriptor (no scikit-image
    dependency required) -- captures cell shape/edge structure.
    """

    resized = cv2.resize(gray_image, (64, 64))

    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )

    return hog.compute(resized).flatten()


def _color_histogram_features(bgr_image: np.ndarray, bins: int = 16) -> np.ndarray:
    """
    Per-channel color histogram -- captures the stain/color signature
    that matters a lot for distinguishing blood cell types (e.g.
    eosinophil's distinctive orange-red granules vs lymphocyte).
    """

    features = []

    for channel in range(3):
        hist = cv2.calcHist([bgr_image], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)

    return np.concatenate(features)


def extract_features(image_path) -> np.ndarray:
    """
    Extract the full feature vector for one image: HOG + color
    histogram, concatenated.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hog_feat = _hog_features(gray)
    color_feat = _color_histogram_features(image)

    return np.concatenate([hog_feat, color_feat])


# =============================================================================
# Dataset -> Feature Matrix
# =============================================================================

def build_feature_matrix(dataset):
    """
    Extract features for every image in a classification-task
    UniversalDataset (whole-image label convention: one BoundingBox
    per image carrying the class).

    Returns
    -------
    tuple
        (X: np.ndarray of shape (n_samples, n_features),
         y: list[str] of class names,
         image_ids: list[str])
    """

    X, y, image_ids = [], [], []

    for image in dataset.images:

        objects = getattr(image, "objects", [])

        if not objects:
            continue

        try:
            features = extract_features(image.image_path)
        except ValueError:
            continue

        X.append(features)
        y.append(objects[0].class_name)
        image_ids.append(image.image_id)

    return np.array(X), y, image_ids


# =============================================================================
# Baseline Model
# =============================================================================

@dataclass
class BaselineClassifierResult:

    accuracy: float
    class_names: List[str]
    predictions: list
    true_labels: list
    image_ids: list


class BaselineClassifier:
    """
    Thin wrapper around an sklearn RandomForestClassifier operating on
    HOG + color-histogram features.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):

        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

        self._trained = False

    def fit(self, dataset) -> None:
        """
        Train on a UniversalDataset's train_set(). Call after
        dataset.assign_splits().
        """

        X, y, _ = build_feature_matrix(dataset.train_set())

        if len(X) == 0:
            raise ValueError("No labeled images found in the training split.")

        self._model.fit(X, y)
        self._trained = True

    def evaluate(self, dataset) -> BaselineClassifierResult:
        """
        Evaluate on a UniversalDataset's test_set() (falls back to
        val_set() if test_set() is empty).
        """

        if not self._trained:
            raise RuntimeError("Call fit() before evaluate().")

        eval_set = dataset.test_set()
        if len(eval_set) == 0:
            eval_set = dataset.val_set()

        X, y_true, image_ids = build_feature_matrix(eval_set)

        if len(X) == 0:
            raise ValueError("No labeled images found in the evaluation split.")

        y_pred = list(self._model.predict(X))

        accuracy = float(np.mean([p == t for p, t in zip(y_pred, y_true)]))

        return BaselineClassifierResult(
            accuracy=accuracy,
            class_names=sorted(set(y_true) | set(y_pred)),
            predictions=y_pred,
            true_labels=y_true,
            image_ids=image_ids,
        )

    def predict_one(self, image_path) -> str:

        if not self._trained:
            raise RuntimeError("Call fit() before predict_one().")

        features = extract_features(image_path).reshape(1, -1)

        return self._model.predict(features)[0]

    @property
    def feature_importances(self) -> Optional[np.ndarray]:

        if not self._trained:
            return None

        return self._model.feature_importances_

    @property
    def sklearn_model(self):
        return self._model
