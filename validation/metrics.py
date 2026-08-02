"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    metrics.py

Version:
    4.0.0

Purpose:
    Dataset Quality Metrics Engine

Description:
    Computes quality metrics for a validated UniversalDataset.

Metrics
-------
✓ Annotation Completeness Score (ACS)
✓ Image Integrity Score (IIS)
✓ Class Consistency Score (CCS)
✓ Bounding Box Validity Score (BVS)
✓ Blood Dataset Quality Index (BDQI)
✓ Overall Dataset Grade

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from .models import (
    ValidationMetrics,
    ValidationResult,
)

from bloodcell.universal_dataset import UniversalDataset


# =============================================================================
# Dataset Metrics Engine
# =============================================================================

class DatasetMetricsEngine:
    """
    Computes quality metrics for a validated dataset.

    The engine uses the ValidationResult produced by the
    validator together with the UniversalDataset statistics.

    This module performs no validation.
    """

    def __init__(self):

        self.metrics = ValidationMetrics()

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self):

        self.metrics.reset()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def compute(
        self,
        dataset: UniversalDataset,
        result: ValidationResult,
    ) -> ValidationMetrics:
        """
        Compute all quality metrics.

        Parameters
        ----------
        dataset : UniversalDataset

        result : ValidationResult

        Returns
        -------
        ValidationMetrics
        """

        self.reset()

        self._annotation_completeness(
            dataset,
            result,
        )

        self._image_integrity(
            dataset,
            result,
        )

        self._class_consistency(
            dataset,
            result,
        )

        self._bounding_box_validity(
            dataset,
            result,
        )

        self._blood_dataset_quality_index()

        self._overall_grade()

        return self.metrics
   
    # -------------------------------------------------------------------------
    # Annotation Completeness Score (ACS)
    # -------------------------------------------------------------------------

    def _annotation_completeness(
        self,
        dataset: UniversalDataset,
        result: ValidationResult,
    ) -> None:
        """
        Annotation Completeness Score (ACS)

        Measures how complete and correct the annotations are.

        Formula

            ACS = 100 - (Errors / Total Objects) × 100
        """

        summary = result.summary

        total_objects = max(
            summary.total_objects,
            1,
        )

        score = 100.0 - (

            (summary.errors / total_objects)

            * 100.0

        )

        self.metrics.annotation_completeness_score = round(

            max(0.0, min(score, 100.0)),

            2,

        )

    # -------------------------------------------------------------------------
    # Image Integrity Score (IIS)
    # -------------------------------------------------------------------------

    def _image_integrity(
        self,
        dataset: UniversalDataset,
        result: ValidationResult,
    ) -> None:
        """
        Image Integrity Score (IIS)

        Measures dataset image integrity.

        Formula

            IIS = 100 - (Errors / Total Images) × 100
        """

        summary = result.summary

        total_images = max(
            summary.total_images,
            1,
        )

        score = 100.0 - (

            (summary.errors / total_images)

            * 100.0

        )

        self.metrics.image_integrity_score = round(

            max(0.0, min(score, 100.0)),

            2,

        )

    # -------------------------------------------------------------------------
    # Class Consistency Score (CCS)
    # -------------------------------------------------------------------------

    def _class_consistency(
        self,
        dataset: UniversalDataset,
        result: ValidationResult,
    ) -> None:
        """
        Class Consistency Score (CCS)

        Measures consistency of class labels.

        Formula

            CCS = 100 - (Warnings / Total Objects) × 100
        """

        summary = result.summary

        total_objects = max(
            summary.total_objects,
            1,
        )

        score = 100.0 - (

            (summary.warnings / total_objects)

            * 100.0

        )

        self.metrics.class_consistency_score = round(

            max(0.0, min(score, 100.0)),

            2,

        )
    # -------------------------------------------------------------------------
    # Bounding Box Validity Score (BVS)
    # -------------------------------------------------------------------------

    def _bounding_box_validity(
        self,
        dataset: UniversalDataset,
        result: ValidationResult,
    ) -> None:
        """
        Bounding Box Validity Score (BVS)

        Measures the overall quality of bounding-box annotations.

        Formula

            BVS = 100 - (Total Issues / Total Objects) × 100

        where

            Total Issues =
                INFO +
                WARNING +
                ERROR +
                CRITICAL
        """

        summary = result.summary

        total_objects = max(
            summary.total_objects,
            1,
        )

        total_issues = (

            summary.info

            + summary.warnings

            + summary.errors

            + summary.critical

        )

        score = 100.0 - (

            (total_issues / total_objects)

            * 100.0

        )

        self.metrics.bounding_box_validity_score = round(

            max(0.0, min(score, 100.0)),

            2,

        )

    # -------------------------------------------------------------------------
    # Blood Dataset Quality Index (BDQI)
    # -------------------------------------------------------------------------

    def _blood_dataset_quality_index(
        self,
    ) -> None:
        """
        Blood Dataset Quality Index (BDQI)

        BDQI is the arithmetic mean of the four quality metrics.

            ACS
            IIS
            CCS
            BVS

        Each metric contributes equally.
        """

        scores = [

            self.metrics.annotation_completeness_score,

            self.metrics.image_integrity_score,

            self.metrics.class_consistency_score,

            self.metrics.bounding_box_validity_score,

        ]

        self.metrics.blood_dataset_quality_index = round(

            sum(scores) / len(scores),

            2,

        )
    # -------------------------------------------------------------------------
    # Overall Grade
    # -------------------------------------------------------------------------

    def _overall_grade(
        self,
    ) -> None:
        """
        Assign an overall quality grade based on the
        Blood Dataset Quality Index (BDQI).

        Grade Scale
        -----------

        A+ : 95 - 100
        A  : 90 - 94.99
        B  : 80 - 89.99
        C  : 70 - 79.99
        D  : 60 - 69.99
        F  : < 60
        """

        score = self.metrics.blood_dataset_quality_index

        if score >= 95:

            grade = "A+"

        elif score >= 90:

            grade = "A"

        elif score >= 80:

            grade = "B"

        elif score >= 70:

            grade = "C"

        elif score >= 60:

            grade = "D"

        else:

            grade = "F"

        self.metrics.overall_grade = grade

# =============================================================================
# Convenience Function
# =============================================================================

def compute_metrics(
    dataset: UniversalDataset,
    result: ValidationResult,
) -> ValidationMetrics:
    """
    Compute quality metrics for a validated dataset.

    Parameters
    ----------
    dataset : UniversalDataset

    result : ValidationResult

    Returns
    -------
    ValidationMetrics
    """

    engine = DatasetMetricsEngine()

    return engine.compute(
        dataset,
        result,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [

    "DatasetMetricsEngine",

    "compute_metrics",

]