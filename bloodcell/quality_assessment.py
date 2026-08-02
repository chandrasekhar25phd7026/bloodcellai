"""
==============================================================
BloodCellAI Quality Assessment Module (Phase 4)
==============================================================

File:
    quality_assessment.py

Description
-----------
Dataset-level quality gate: run blur, contrast, brightness, entropy,
and corruption checks across every image in a (already built)
UniversalDataset, and produce one aggregate report a researcher can
inspect *before* handing the dataset to a training pipeline.

Design notes
------------
- Deliberately dependency-light: bloodcell should not *require* the
  full preprocessing package just to answer "is this dataset clean
  enough to train on?" If Phase 3's `enable_preprocessing()` was used
  during the build, this reuses the quality metrics already computed
  and cached in each image's `metadata["preprocessing"]` (no
  recomputation). If not, it falls back to a small, self-contained
  set of checks implemented directly here with OpenCV/NumPy --
  correct, but a lighter-weight equivalent of the full preprocessing
  QualityTransform.
- Ties together Phase 1 (UniversalDataset.filter()) and Phase 3
  (per-image metadata) into one "get me the clean subset to train
  on" call.

Author:
    BloodCellAI Project
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# =============================================================================
# Thresholds
# =============================================================================

@dataclass
class QualityThresholds:
    """
    Same threshold names/defaults as preprocessing.QualityConfig, kept
    independent so bloodcell doesn't require importing preprocessing
    just to define its own defaults.
    """

    minimum_brightness: float = 20.0
    maximum_brightness: float = 235.0
    minimum_contrast: float = 20.0
    minimum_sharpness: float = 50.0
    maximum_blur: float = 200.0
    minimum_entropy: float = 3.0
    minimum_quality_score: float = 70.0

    @classmethod
    def from_preprocessing_config(cls, quality_config) -> "QualityThresholds":
        """
        Build thresholds from an existing
        preprocessing.preprocessing_config.QualityConfig instance, so
        the two stay consistent if the caller already has one.
        """

        return cls(
            minimum_brightness=quality_config.minimum_brightness,
            maximum_brightness=quality_config.maximum_brightness,
            minimum_contrast=quality_config.minimum_contrast,
            minimum_sharpness=quality_config.minimum_sharpness,
            maximum_blur=quality_config.maximum_blur,
            minimum_entropy=quality_config.minimum_entropy,
            minimum_quality_score=quality_config.minimum_quality_score,
        )


# =============================================================================
# Per-Image Result
# =============================================================================

@dataclass
class ImageQualityResult:

    image_id: str
    image_path: str
    corrupted: bool = False
    error: str = ""
    brightness: float = 0.0
    contrast: float = 0.0
    blur_score: float = 0.0
    entropy: float = 0.0
    quality_score: float = 0.0
    passed: bool = True
    issues: list = field(default_factory=list)
    source: str = "computed"  # "computed" or "cached" (from Phase 3 metadata)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "corrupted": self.corrupted,
            "error": self.error,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "blur_score": self.blur_score,
            "entropy": self.entropy,
            "quality_score": self.quality_score,
            "passed": self.passed,
            "issues": self.issues,
            "source": self.source,
        }


# =============================================================================
# Native (lightweight, dependency-light) Per-Image Quality Check
# =============================================================================

def assess_image_quality(
    image_path,
    thresholds: Optional[QualityThresholds] = None,
) -> ImageQualityResult:
    """
    Run blur, contrast, brightness, entropy, and corruption checks on
    a single image file, without requiring the preprocessing package.

    This is a lighter-weight equivalent of
    preprocessing.quality.QualityTransform -- same measurements
    (Laplacian-variance blur, std-based contrast, mean brightness,
    histogram entropy), same threshold names/defaults, computed
    directly with OpenCV/NumPy so bloodcell can answer "is this image
    OK" on its own.
    """

    thresholds = thresholds or QualityThresholds()

    image_path = str(image_path)
    image_id = Path(image_path).stem

    result = ImageQualityResult(image_id=image_id, image_path=image_path)

    # ---------------------------------------------------------------
    # Corruption check
    # ---------------------------------------------------------------

    if not Path(image_path).exists():
        result.corrupted = True
        result.error = f"Image file does not exist: {image_path}"
        result.passed = False
        result.issues.append("missing_file")
        return result

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        result.corrupted = True
        result.error = f"Unable to read image (corrupted or unsupported format): {image_path}"
        result.passed = False
        result.issues.append("corrupted")
        return result

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    result.brightness = float(np.mean(gray))
    result.contrast = float(np.std(gray))

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    result.blur_score = float(laplacian.var())

    histogram, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    histogram = histogram[histogram > 0]
    result.entropy = float(-np.sum(histogram * np.log2(histogram))) if len(histogram) else 0.0

    # ---------------------------------------------------------------
    # Composite quality score (same approach as
    # preprocessing.quality.QualityTransform._quality_score)
    # ---------------------------------------------------------------

    scores = []

    if thresholds.minimum_brightness <= result.brightness <= thresholds.maximum_brightness:
        scores.append(100.0)
    else:
        distance = min(
            abs(result.brightness - thresholds.minimum_brightness),
            abs(result.brightness - thresholds.maximum_brightness),
        )
        scores.append(max(0.0, 100.0 - distance))

    scores.append(
        min(100.0, 100.0 * result.contrast / thresholds.minimum_contrast)
        if thresholds.minimum_contrast > 0 else 100.0
    )
    scores.append(
        min(100.0, 100.0 * result.blur_score / thresholds.maximum_blur)
        if thresholds.maximum_blur > 0 else 100.0
    )
    scores.append(
        min(100.0, 100.0 * result.entropy / thresholds.minimum_entropy)
        if thresholds.minimum_entropy > 0 else 100.0
    )

    result.quality_score = round(sum(scores) / len(scores), 2)

    # ---------------------------------------------------------------
    # Threshold checks -> issues
    # ---------------------------------------------------------------

    if result.brightness < thresholds.minimum_brightness:
        result.issues.append("brightness_too_low")
    if result.brightness > thresholds.maximum_brightness:
        result.issues.append("brightness_too_high")
    if result.contrast < thresholds.minimum_contrast:
        result.issues.append("contrast_too_low")
    if result.blur_score < thresholds.maximum_blur:
        result.issues.append("blurry")
    if result.entropy < thresholds.minimum_entropy:
        result.issues.append("entropy_too_low")
    if result.quality_score < thresholds.minimum_quality_score:
        result.issues.append("quality_score_below_threshold")

    result.passed = len(result.issues) == 0

    return result


def _result_from_cached_metadata(image_id, image_path, cached: dict) -> ImageQualityResult:
    """
    Build an ImageQualityResult from the metadata["preprocessing"]
    block already attached during Phase 3's build_from_info(
    ..., preprocessing_manager=...) -- avoids recomputing quality
    metrics that were already computed once during dataset creation.
    """

    if not cached.get("passed", True) and "error" in cached:
        return ImageQualityResult(
            image_id=image_id,
            image_path=image_path,
            corrupted=True,
            error=cached["error"],
            passed=False,
            issues=["corrupted"],
            source="cached",
        )

    metrics = cached.get("quality_metrics", {})

    issues = list(cached.get("warnings", []))

    return ImageQualityResult(
        image_id=image_id,
        image_path=image_path,
        brightness=metrics.get("brightness", 0.0),
        contrast=metrics.get("contrast", 0.0),
        blur_score=metrics.get("blur_score", 0.0),
        entropy=metrics.get("entropy", 0.0),
        quality_score=cached.get("quality_score", 0.0),
        passed=cached.get("passed", True),
        issues=issues,
        source="cached",
    )


# =============================================================================
# Dataset-Level Report
# =============================================================================

@dataclass
class DatasetQualityReport:

    total_images: int = 0
    passed: int = 0
    failed: int = 0
    corrupted: int = 0
    mean_quality_score: float = 0.0
    results: list = field(default_factory=list)  # list[ImageQualityResult]

    @property
    def pass_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return round(100.0 * self.passed / self.total_images, 2)

    @property
    def failed_image_ids(self) -> list:
        return [r.image_id for r in self.results if not r.passed]

    def issue_counts(self) -> dict:
        """
        How many images were flagged for each issue type -- useful
        for deciding which threshold is actually the bottleneck
        before training (e.g. "80% of failures are 'blurry'").
        """

        counts: dict = {}

        for result in self.results:
            for issue in result.issues:
                counts[issue] = counts.get(issue, 0) + 1

        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "passed": self.passed,
            "failed": self.failed,
            "corrupted": self.corrupted,
            "pass_rate": self.pass_rate,
            "mean_quality_score": self.mean_quality_score,
            "issue_counts": self.issue_counts(),
            "failed_image_ids": self.failed_image_ids,
        }

    def save_json(self, path) -> None:
        import json

        with open(path, "w") as f:
            json.dump(
                {**self.to_dict(), "results": [r.to_dict() for r in self.results]},
                f,
                indent=2,
            )

    def summary_text(self) -> str:

        lines = [
            "=" * 70,
            "BloodCellAI Dataset Quality Report",
            "=" * 70,
            f"Total images   : {self.total_images}",
            f"Passed         : {self.passed} ({self.pass_rate}%)",
            f"Failed         : {self.failed}",
            f"Corrupted      : {self.corrupted}",
            f"Mean quality   : {self.mean_quality_score}",
            "",
            "Issue breakdown:",
        ]

        for issue, count in self.issue_counts().items():
            lines.append(f"  {issue:30} {count}")

        return "\n".join(lines)


# =============================================================================
# Public API
# =============================================================================

def assess_dataset_quality(
    dataset,
    thresholds: Optional[QualityThresholds] = None,
    use_cached: bool = True,
) -> DatasetQualityReport:
    """
    Run the quality gate across every image in a UniversalDataset.

    Parameters
    ----------
    dataset : bloodcell.universal_dataset.UniversalDataset
        An already-built dataset.

    thresholds : QualityThresholds, optional
        Defaults applied if not given.

    use_cached : bool
        If True (default) and an image already has
        metadata["preprocessing"] (from Phase 3's
        enable_preprocessing()), reuse it instead of recomputing.
        Set False to force fresh computation for every image
        regardless of cached metadata.

    Returns
    -------
    DatasetQualityReport
    """

    thresholds = thresholds or QualityThresholds()

    results = []

    for image in dataset.images:

        cached = image.metadata.get("preprocessing") if use_cached else None

        if cached is not None:
            result = _result_from_cached_metadata(
                image.image_id, image.image_path, cached
            )
        else:
            result = assess_image_quality(image.image_path, thresholds)

        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    corrupted = sum(1 for r in results if r.corrupted)
    mean_score = (
        round(sum(r.quality_score for r in results) / total, 2)
        if total else 0.0
    )

    return DatasetQualityReport(
        total_images=total,
        passed=passed,
        failed=total - passed,
        corrupted=corrupted,
        mean_quality_score=mean_score,
        results=results,
    )


def filter_to_quality_passing(dataset, report: DatasetQualityReport):
    """
    Return a new UniversalDataset (via UniversalDataset.filter(),
    Phase 1) containing only images that passed the quality gate --
    the "clean subset to actually train on" this module exists to
    produce.
    """

    passing_ids = {r.image_id for r in report.results if r.passed}

    return dataset.filter(lambda img: img.image_id in passing_ids)
