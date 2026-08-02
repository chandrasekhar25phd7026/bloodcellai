"""
BloodCellAI Framework
Validation Engine

File:
    validator.py

Description:
    Central validation engine for validating a UniversalDataset.
    Runs all rule modules (dataset/image/object/bbox), collects
    ValidationIssue records, computes DatasetStatistics, and
    computes quality metrics (including BDQI) via
    ValidationMetricsCalculator.

Author:
    BloodCellAI Project

Version:
    1.1.0
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .dataset_rules import validate_dataset
from .image_rules import validate_images
from .object_rules import validate_objects
from .bbox_rules import validate_bounding_boxes
from .statistics import DatasetStatistics
from .metrics import ValidationMetricsCalculator

from .models import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationCategory,
)

logger = logging.getLogger(__name__)


class DatasetValidator:
    """
    Central validation engine.

    Responsibilities
    ----------------
    1. Validate a UniversalDataset (dataset/image/object/bbox rules)
    2. Collect validation issues
    3. Compute dataset statistics
    4. Compute quality metrics, including the BloodCell Dataset
       Quality Index (BDQI)
    5. Return a single ValidationResult
    """

    def __init__(self) -> None:

        self._issue_counter = 0
        self._issues: list[ValidationIssue] = []

        # image_path -> dataset name, built fresh per validate() call so
        # that rule modules (which only receive `image`/`image_path`) can
        # still have their issues attributed to the correct dataset.
        self._path_to_dataset: dict[str, str] = {}

    #######################################################################
    # Public API
    #######################################################################

    def validate(self, dataset) -> ValidationResult:
        """
        Validate a UniversalDataset.

        Parameters
        ----------
        dataset : UniversalDataset

        Returns
        -------
        ValidationResult
        """

        logger.info("Starting dataset validation...")

        start_time = time.time()

        self._reset(dataset)

        # ----------------------------------------------------------
        # Rule Modules
        # ----------------------------------------------------------

        validate_dataset(self, dataset)
        validate_images(self, dataset)
        validate_objects(self, dataset)
        validate_bounding_boxes(self, dataset)

        # ----------------------------------------------------------
        # Assemble Result
        # ----------------------------------------------------------

        result = ValidationResult()
        result.issues = self._issues

        self._finalize_summary(result, start_time)
        self._finalize_statistics(result, dataset)
        self._finalize_metrics(result)

        logger.info("Validation finished.")

        return result

    #######################################################################
    # Internal Helpers
    #######################################################################

    def _reset(self, dataset) -> None:
        """
        Reset validator state for a new validation run.
        """

        self._issue_counter = 0
        self._issues.clear()

        self._path_to_dataset = {
            getattr(image, "image_path", None): getattr(image, "dataset", None)
            for image in getattr(dataset, "images", []) or []
        }

    def _generate_issue_id(self) -> str:
        """
        Generate unique issue id.

        Returns
        -------
        str
        """

        self._issue_counter += 1

        return f"VAL-{self._issue_counter:06d}"

    #######################################################################
    # Issue Management
    #######################################################################

    def _add_issue(
        self,
        severity: ValidationSeverity,
        category: ValidationCategory,
        message: str,
        recommendation: str = "",
        dataset: Optional[str] = None,
        image_path: Optional[str] = None,
        object_index: Optional[int] = None,
    ) -> None:
        """
        Add a validation issue.

        If `dataset` is not supplied but `image_path` is, the dataset
        name is resolved from the image-path index built in `_reset`,
        so every issue can be attributed to a dataset even though the
        per-rule check functions only see the image, not the dataset.
        """

        if dataset is None and image_path is not None:
            dataset = self._path_to_dataset.get(image_path)

        issue = ValidationIssue(
            issue_id=self._generate_issue_id(),
            severity=severity,
            category=category,
            dataset=dataset or "",
            image_path=image_path or "",
            object_index=object_index,
            message=message,
            recommendation=recommendation,
        )

        self._issues.append(issue)

    def _add_error(
        self,
        category: ValidationCategory,
        message: str,
        recommendation: str = "",
        dataset: Optional[str] = None,
        image_path: Optional[str] = None,
        object_index: Optional[int] = None,
    ) -> None:

        self._add_issue(
            ValidationSeverity.ERROR,
            category,
            message,
            recommendation,
            dataset,
            image_path,
            object_index,
        )

    def _add_warning(
        self,
        category: ValidationCategory,
        message: str,
        recommendation: str = "",
        dataset: Optional[str] = None,
        image_path: Optional[str] = None,
        object_index: Optional[int] = None,
    ) -> None:

        self._add_issue(
            ValidationSeverity.WARNING,
            category,
            message,
            recommendation,
            dataset,
            image_path,
            object_index,
        )

    def _add_info(
        self,
        category: ValidationCategory,
        message: str,
        recommendation: str = "",
        dataset: Optional[str] = None,
        image_path: Optional[str] = None,
        object_index: Optional[int] = None,
    ) -> None:

        self._add_issue(
            ValidationSeverity.INFO,
            category,
            message,
            recommendation,
            dataset,
            image_path,
            object_index,
        )

    #######################################################################
    # Summary / Statistics / Metrics
    #######################################################################

    def _finalize_summary(
        self,
        result: ValidationResult,
        start_time: float,
    ) -> None:
        """
        Populate validation summary.
        """

        summary = result.summary

        summary.total_issues = len(self._issues)

        summary.info = sum(
            issue.severity == ValidationSeverity.INFO
            for issue in self._issues
        )

        summary.warnings = sum(
            issue.severity == ValidationSeverity.WARNING
            for issue in self._issues
        )

        summary.errors = sum(
            issue.severity == ValidationSeverity.ERROR
            for issue in self._issues
        )

        summary.critical = sum(
            issue.severity == ValidationSeverity.CRITICAL
            for issue in self._issues
        )

        summary.passed = (
            summary.errors == 0
            and summary.critical == 0
        )

        summary.validation_time = (
            time.time() - start_time
        )

    def _finalize_statistics(self, result: ValidationResult, dataset) -> None:
        """
        Populate dataset statistics using DatasetStatistics.
        """

        stats = DatasetStatistics().compute(dataset).to_dict()

        result.statistics.total_images = stats["total_images"]
        result.statistics.total_objects = stats["total_objects"]
        result.statistics.number_of_classes = stats["number_of_classes"]
        result.statistics.objects_per_image = stats["objects_per_image"]
        result.statistics.average_width = stats["average_width"]
        result.statistics.average_height = stats["average_height"]
        result.statistics.dataset_counts = stats["dataset_counts"]
        result.statistics.class_counts = stats["class_counts"]

    def _finalize_metrics(self, result: ValidationResult) -> None:
        """
        Compute quality metrics (including BDQI) from the finalized
        summary/issues via ValidationMetricsCalculator.
        """

        calculator = ValidationMetricsCalculator().compute(result)

        result.metrics.annotation_completeness_score = (
            calculator.annotation_completeness_score
        )
        result.metrics.image_integrity_score = (
            calculator.image_integrity_score
        )
        result.metrics.class_consistency_score = (
            calculator.class_consistency_score
        )
        result.metrics.bounding_box_validity_score = (
            calculator.bounding_box_validity_score
        )
        result.metrics.bdqi = calculator.bdqi
