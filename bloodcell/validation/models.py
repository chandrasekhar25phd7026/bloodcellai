"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    models.py

Version:
    1.1.0

Status:
    Stable

Purpose:
    Core data models used by the BloodCellAI validation package.

Author:
    Sekhar Muthangi

Project:
    BloodCellAI
===============================================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# =============================================================================
# Validation Severity
# =============================================================================

class ValidationSeverity(Enum):
    """Severity level of a validation issue."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# Validation Category
# =============================================================================

class ValidationCategory(Enum):
    """Category of validation."""

    DATASET = "Dataset"
    IMAGE = "Image"
    OBJECT = "Object"
    BOUNDING_BOX = "Bounding Box"
    CLASS = "Class"
    METADATA = "Metadata"
    STATISTICS = "Statistics"
    METRICS = "Metrics"


# =============================================================================
# Validation Issue
# =============================================================================

@dataclass(slots=True)
class ValidationIssue:
    """
    Represents one validation issue.
    """

    issue_id: str
    severity: ValidationSeverity
    category: ValidationCategory
    dataset: str

    image_path: str = ""
    object_index: Optional[int] = None

    message: str = ""
    recommendation: str = ""

    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Validation Summary
# =============================================================================

@dataclass(slots=True)
class ValidationSummary:
    """
    Summary of validation results.
    """

    total_issues: int = 0

    info: int = 0
    warnings: int = 0
    errors: int = 0
    critical: int = 0

    passed: bool = True

    validation_time: float = 0.0

    def reset(self):

        self.total_issues = 0
        self.info = 0
        self.warnings = 0
        self.errors = 0
        self.critical = 0
        self.passed = True
        self.validation_time = 0.0

    def to_dict(self):

        return asdict(self)


# =============================================================================
# Validation Statistics
# =============================================================================

@dataclass(slots=True)
class ValidationStatistics:
    """
    Dataset statistics computed during validation.
    """

    # Basic Counts
    total_images: int = 0
    total_objects: int = 0

    # Derived Statistics
    number_of_classes: int = 0
    objects_per_image: float = 0.0

    average_width: float = 0.0
    average_height: float = 0.0

    # Distribution
    dataset_counts: Dict[str, int] = field(default_factory=dict)
    class_counts: Dict[str, int] = field(default_factory=dict)

    def reset(self):

        self.total_images = 0
        self.total_objects = 0

        self.number_of_classes = 0
        self.objects_per_image = 0.0

        self.average_width = 0.0
        self.average_height = 0.0

        self.dataset_counts.clear()
        self.class_counts.clear()

    def to_dict(self):

        return asdict(self)


# =============================================================================
# Validation Metrics
# =============================================================================

@dataclass(slots=True)
class ValidationMetrics:
    """
    Dataset quality metrics.
    """

    annotation_completeness_score: float = 0.0

    image_integrity_score: float = 0.0

    class_consistency_score: float = 0.0

    bounding_box_validity_score: float = 0.0

    bdqi: float = 0.0

    def reset(self):

        self.annotation_completeness_score = 0.0
        self.image_integrity_score = 0.0
        self.class_consistency_score = 0.0
        self.bounding_box_validity_score = 0.0
        self.bdqi = 0.0

    def to_dict(self):

        return asdict(self)


# =============================================================================
# Validation Result
# =============================================================================

@dataclass(slots=True)
class ValidationResult:
    """
    Complete validation output.
    """

    issues: List[ValidationIssue] = field(default_factory=list)

    summary: ValidationSummary = field(default_factory=ValidationSummary)

    statistics: ValidationStatistics = field(default_factory=ValidationStatistics)

    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)

    def reset(self):

        self.issues.clear()

        self.summary.reset()
        self.statistics.reset()
        self.metrics.reset()

    def to_dict(self):

        return {
            "summary": self.summary.to_dict(),
            "statistics": self.statistics.to_dict(),
            "metrics": self.metrics.to_dict(),
            "issues": [
                {
                    "issue_id": issue.issue_id,
                    "severity": issue.severity.name,
                    "category": issue.category.name,
                    "dataset": issue.dataset,
                    "image_path": issue.image_path,
                    "object_index": issue.object_index,
                    "message": issue.message,
                    "recommendation": issue.recommendation,
                    "timestamp": issue.timestamp.isoformat(),
                }
                for issue in self.issues
            ],
        }