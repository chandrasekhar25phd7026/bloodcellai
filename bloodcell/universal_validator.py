"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

Module:
    universal_validator.py

Description:
    Enterprise-grade validation framework for validating UniversalDataset
    objects produced by the BloodCellAI Universal Builder.

This module validates the standardized UniversalDataset representation
rather than dataset-specific annotation formats.

Future Versions
---------------
v1.0.0 : Core Architecture
v1.1.0 : Structural Validation
v1.2.0 : Image Validation
v1.3.0 : Annotation Validation
v1.4.0 : Statistics Engine
v1.5.0 : Quality Metrics (ACS, IIS, CCS, BVS, BDQI)

Author:
    Sekhar Muthangi

Project:
    BloodCellAI

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from .universal_dataset import UniversalDataset


# =============================================================================
# Validation Severity
# =============================================================================

class ValidationSeverity(Enum):
    """
    Severity level of a validation issue.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# Validation Category
# =============================================================================

class ValidationCategory(Enum):
    """
    Category of validation issue.
    """

    STRUCTURE = "Structure"
    IMAGE = "Image"
    ANNOTATION = "Annotation"
    CLASS = "Class"
    BOUNDING_BOX = "BoundingBox"
    DUPLICATE = "Duplicate"
    METADATA = "Metadata"
    DATASET = "Dataset"


# =============================================================================
# Validation Issue
# =============================================================================

@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.
    """

    severity: ValidationSeverity
    category: ValidationCategory
    dataset: str
    image_path: str
    message: str
    recommendation: str = ""


# =============================================================================
# Validation Summary
# =============================================================================

@dataclass
class ValidationSummary:
    """
    High-level validation summary.
    """

    total_images: int = 0
    total_objects: int = 0

    errors: int = 0
    warnings: int = 0

    validation_time: Optional[datetime] = None


# =============================================================================
# Validation Statistics
# =============================================================================

@dataclass
class ValidationStatistics:
    """
    Stores dataset statistics collected during validation.
    """

    class_counts: Dict[str, int] = field(default_factory=dict)
    dataset_counts: Dict[str, int] = field(default_factory=dict)

    average_objects_per_image: float = 0.0

    minimum_width: int = 0
    maximum_width: int = 0

    minimum_height: int = 0
    maximum_height: int = 0


# =============================================================================
# Validation Metrics
# =============================================================================

@dataclass
class ValidationMetrics:
    """
    Dataset quality metrics.

    These values will be calculated in later versions.
    """

    annotation_completeness_score: float = 0.0
    image_integrity_score: float = 0.0
    class_consistency_score: float = 0.0
    bounding_box_validity_score: float = 0.0
    bdqi: float = 0.0


# =============================================================================
# Validation Result
# =============================================================================

@dataclass
class ValidationResult:
    """
    Complete validation output.
    """

    summary: ValidationSummary = field(default_factory=ValidationSummary)

    statistics: ValidationStatistics = field(default_factory=ValidationStatistics)

    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)

    issues: List[ValidationIssue] = field(default_factory=list)


# =============================================================================
# Dataset Validator
# =============================================================================

class DatasetValidator:
    """
    BloodCellAI Universal Dataset Validator.

    This class validates UniversalDataset objects.
    """

    def __init__(self) -> None:

        self.version = "1.0.0"

        self.name = "BloodCellAI Dataset Validation Engine"

    def validate(self, dataset: UniversalDataset) -> ValidationResult:
        """
        Validate a UniversalDataset.

        Parameters
        ----------
        dataset : UniversalDataset
            Dataset to validate.

        Returns
        -------
        ValidationResult
            Validation result object.
        """

        if not isinstance(dataset, UniversalDataset):

            raise TypeError(
                "Expected a UniversalDataset object."
            )

        result = ValidationResult()

        result.summary.total_images = len(dataset.images)

        result.summary.total_objects = sum(
            len(image.objects)
            for image in dataset.images
        )

        result.summary.validation_time = datetime.now()

        result.statistics.class_counts = dict(dataset.class_counts)

        result.statistics.dataset_counts = dict(dataset.dataset_counts)

        if len(dataset.images) > 0:

            result.statistics.average_objects_per_image = (
                result.summary.total_objects /
                result.summary.total_images
            )

        return result