"""
==============================================================
BloodCellAI Dataset Quality Gate
==============================================================

File:
    quality_gate.py

Description
-----------
Dedicated quality-assessment module: runs blur, contrast, brightness,
entropy, and corruption checks across an ENTIRE UniversalDataset (not
just one image), and produces both a per-image quality report and a
dataset-level summary -- this is the "before training" gate: run it
once a dataset is built, then hand `.filter_passing()`'s result to
the training pipeline instead of the raw, unfiltered dataset.

Relationship to preprocessing.QualityTransform
-----------------------------------------------
QualityTransform (in the preprocessing package) computes these same
metrics for ONE image, as part of the broader preprocessing pipeline
(resize/normalize/etc). This module is the dataset-level counterpart:
it aggregates per-image results (reusing an image's
`metadata["preprocessing"]` block if UniversalBuilder.enable_preprocessing()
was already used during dataset creation -- see universal_builder.py)
or computes them fresh via preprocessing.PreprocessingManager if not,
and turns them into a dataset-wide, trainer-facing decision: which
images are actually fit to train on.

Design notes
------------
The `preprocessing` package is treated as an optional, soft
dependency here (imported lazily, inside methods) so that a bare
`bloodcell` install can still build and inspect datasets without
requiring `preprocessing`/`transforms` to be installed -- consistent
with how the four packages in this project are kept independently
installable.

Author:
    BloodCellAI Project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Per-Image Quality Record
# =============================================================================

@dataclass
class ImageQualityRecord:
    """
    Result of assessing one image.
    """

    image_id: str
    image_path: str
    dataset: str

    corrupted: bool = False
    corruption_reason: str = ""

    quality_score: Optional[float] = None
    passed: bool = True
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:

        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "dataset": self.dataset,
            "corrupted": self.corrupted,
            "corruption_reason": self.corruption_reason,
            "quality_score": self.quality_score,
            "passed": self.passed,
            "reasons": self.reasons,
        }


# =============================================================================
# Dataset Quality Report
# =============================================================================

@dataclass
class DatasetQualityReport:
    """
    Dataset-wide quality summary produced by DatasetQualityGate.assess().
    """

    total_images: int = 0
    corrupted_count: int = 0
    passed_count: int = 0
    failed_count: int = 0

    mean_quality_score: Optional[float] = None
    min_quality_score: Optional[float] = None
    max_quality_score: Optional[float] = None

    failure_reason_counts: dict = field(default_factory=dict)

    records: list = field(default_factory=list)

    @property
    def pass_rate(self) -> float:

        if self.total_images == 0:
            return 0.0

        return round(self.passed_count / self.total_images, 4)

    def to_dict(self) -> dict:

        return {
            "total_images": self.total_images,
            "corrupted_count": self.corrupted_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate": self.pass_rate,
            "mean_quality_score": self.mean_quality_score,
            "min_quality_score": self.min_quality_score,
            "max_quality_score": self.max_quality_score,
            "failure_reason_counts": self.failure_reason_counts,
        }

    def summary_text(self) -> str:

        lines = [
            "=" * 70,
            "Dataset Quality Report",
            "=" * 70,
            f"Total images       : {self.total_images}",
            f"Corrupted          : {self.corrupted_count}",
            f"Passed             : {self.passed_count} ({self.pass_rate:.1%})",
            f"Failed             : {self.failed_count}",
        ]

        if self.mean_quality_score is not None:
            lines.append(
                f"Quality score      : mean={self.mean_quality_score:.2f} "
                f"min={self.min_quality_score:.2f} max={self.max_quality_score:.2f}"
            )

        if self.failure_reason_counts:
            lines.append("")
            lines.append("Failure reasons:")
            for reason, count in sorted(
                self.failure_reason_counts.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  {count:4}  {reason}")

        return "\n".join(lines)


# =============================================================================
# Dataset Quality Gate
# =============================================================================

class DatasetQualityGate:
    """
    Runs corruption + quality checks across an entire UniversalDataset
    and decides which images are fit to train on.
    """

    def __init__(
        self,
        minimum_quality_score: float = 50.0,
        preprocessing_config=None,
    ):
        """
        Parameters
        ----------
        minimum_quality_score : float
            Images scoring below this are marked failed (not
            necessarily corrupted -- e.g. very blurry or very dark
            images still "load" fine but shouldn't be trained on).

        preprocessing_config : preprocessing.PreprocessingConfig, optional
            Used only for images that don't already have a
            `metadata["preprocessing"]` block (i.e. the dataset was
            built without UniversalBuilder.enable_preprocessing()).
            Defaults to a fresh PreprocessingConfig() when needed and
            not given.
        """

        self.minimum_quality_score = minimum_quality_score
        self._preprocessing_config = preprocessing_config
        self._manager = None

    def _get_manager(self):
        """
        Lazily construct a PreprocessingManager, only if actually
        needed (i.e. some image lacks precomputed quality metadata).
        Soft dependency: raises a clear error if `preprocessing` isn't
        installed, rather than failing at import time for users who
        only need bloodcell + already-preprocessed datasets.
        """

        if self._manager is None:

            try:
                from preprocessing.preprocessing_config import PreprocessingConfig
                from preprocessing.preprocessing_manager import PreprocessingManager
            except ImportError as exc:
                raise ImportError(
                    "DatasetQualityGate needs to compute quality metrics "
                    "for one or more images that don't already have a "
                    "metadata['preprocessing'] block (i.e. the dataset "
                    "was built without UniversalBuilder.enable_preprocessing()). "
                    "This requires the 'preprocessing' package to be "
                    "installed alongside 'bloodcell'."
                ) from exc

            config = self._preprocessing_config or PreprocessingConfig()

            self._manager = PreprocessingManager(config)

        return self._manager

    # -------------------------------------------------------------------
    # Per-image assessment
    # -------------------------------------------------------------------

    def _assess_image(self, image) -> ImageQualityRecord:

        record = ImageQualityRecord(
            image_id=getattr(image, "image_id", ""),
            image_path=str(getattr(image, "image_path", "")),
            dataset=getattr(image, "dataset", ""),
        )

        # First: does the file even exist / open at all? This is
        # checked directly (not just via preprocessing metadata) so
        # the gate still catches missing files even for datasets that
        # were built with a stale metadata block, or without
        # preprocessing at all.
        path = Path(record.image_path)

        if not path.exists():

            record.corrupted = True
            record.corruption_reason = "File does not exist."
            record.passed = False
            record.reasons.append("corrupted")

            return record

        existing = image.metadata.get("preprocessing") if hasattr(image, "metadata") else None

        if existing is not None:

            # Reuse metadata already computed during dataset creation
            # (UniversalBuilder.enable_preprocessing()) instead of
            # recomputing it.
            if not existing.get("passed", True) and "error" in existing:

                record.corrupted = True
                record.corruption_reason = existing["error"]
                record.passed = False
                record.reasons.append("corrupted")

                return record

            record.quality_score = existing.get("quality_score")
            record.reasons.extend(existing.get("warnings", []))

        else:

            # No precomputed metadata -- compute it now.
            try:

                manager = self._get_manager()

                result = manager.preprocess_image(str(path))

                record.quality_score = result.quality_metrics.quality_score
                record.reasons.extend(result.warnings)

            except ImportError:
                raise

            except Exception as exc:

                record.corrupted = True
                record.corruption_reason = str(exc)
                record.passed = False
                record.reasons.append("corrupted")

                return record

        if (
            record.quality_score is not None
            and record.quality_score < self.minimum_quality_score
        ):
            record.passed = False
            record.reasons.append(
                f"quality_score {record.quality_score:.2f} below "
                f"minimum {self.minimum_quality_score:.2f}"
            )

        return record

    # -------------------------------------------------------------------
    # Dataset-wide assessment
    # -------------------------------------------------------------------

    def assess(self, dataset) -> DatasetQualityReport:
        """
        Assess every image in a UniversalDataset.

        Parameters
        ----------
        dataset : bloodcell.universal_dataset.UniversalDataset

        Returns
        -------
        DatasetQualityReport
        """

        report = DatasetQualityReport()

        scores = []
        reason_counts: dict = {}

        for image in dataset:

            record = self._assess_image(image)

            report.records.append(record)
            report.total_images += 1

            if record.corrupted:
                report.corrupted_count += 1

            if record.passed:
                report.passed_count += 1
            else:
                report.failed_count += 1

            if record.quality_score is not None:
                scores.append(record.quality_score)

            for reason in record.reasons:
                # Bucket free-text quality_score reasons under one
                # label so the summary stays readable, while keeping
                # "corrupted" and specific warning strings distinct.
                label = (
                    reason if reason == "corrupted"
                    or not reason.startswith("quality_score")
                    else "quality_score below minimum"
                )
                reason_counts[label] = reason_counts.get(label, 0) + 1

        if scores:
            report.mean_quality_score = round(sum(scores) / len(scores), 2)
            report.min_quality_score = round(min(scores), 2)
            report.max_quality_score = round(max(scores), 2)

        report.failure_reason_counts = reason_counts

        logger.info(
            "Quality assessment complete: %d/%d images passed (%.1f%%), "
            "%d corrupted.",
            report.passed_count, report.total_images,
            report.pass_rate * 100, report.corrupted_count,
        )

        return report

    # -------------------------------------------------------------------
    # Filtering for training
    # -------------------------------------------------------------------

    def filter_passing(self, dataset):
        """
        Assess `dataset` and return a NEW UniversalDataset containing
        only the images that passed -- this is the actual "quality
        gate before training" step: hand this result to the training
        pipeline instead of the raw dataset.

        Returns
        -------
        tuple
            (filtered_dataset, report)
        """

        report = self.assess(dataset)

        passed_ids = {
            record.image_id for record in report.records if record.passed
        }

        filtered = dataset.filter(
            lambda img: getattr(img, "image_id", None) in passed_ids
        )

        filtered.set_metadata("quality_gate_report", report.to_dict())

        return filtered, report
