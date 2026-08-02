"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

models.py

Author : BloodCellAI Research Framework
Version: 3.0.0

Core data models shared by the BloodCellAI Dataset Validation Engine.

Used by:

    • validator.py
    • statistics.py
    • metrics.py
    • reports.py
    • dataset_rules.py
    • image_rules.py
    • object_rules.py
    • bbox_rules.py

These models are intentionally lightweight and contain no validation logic.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Validation Severity
# =============================================================================

class ValidationSeverity(Enum):
    """
    Severity level assigned to a validation issue.
    """

    INFO = "INFO"

    WARNING = "WARNING"

    ERROR = "ERROR"

    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Validation Category
# =============================================================================

class ValidationCategory(Enum):
    """
    Logical category describing where a validation issue occurred.
    """

    DATASET = "Dataset"

    IMAGE = "Image"

    OBJECT = "Object"

    BOUNDING_BOX = "Bounding Box"

    CLASS = "Class"

    FILE = "File"

    DIRECTORY = "Directory"

    DUPLICATE = "Duplicate"

    METADATA = "Metadata"

    STATISTICS = "Statistics"

    METRICS = "Metrics"

    QUALITY = "Quality"

    UNKNOWN = "Unknown"

    def __str__(self) -> str:
        return self.value
# =============================================================================
# Validation Metadata
# =============================================================================

@dataclass(slots=True)
class ValidationMetadata:
    """
    Metadata describing a validation session.
    """

    framework_name: str = "BloodCellAI Dataset Validation Engine"

    framework_version: str = "3.0.0"

    validator_name: str = "BDVE"

    validator_version: str = "3.0.0"

    dataset_name: str = ""

    dataset_version: str = ""

    dataset_path: str = ""

    report_directory: str = ""

    started_at: datetime = field(
        default_factory=datetime.now
    )

    completed_at: Optional[datetime] = None

    execution_time: float = 0.0

    user_metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------

    @property
    def dataset_directory(self) -> Optional[Path]:

        if self.dataset_path:
            return Path(self.dataset_path)

        return None

    @property
    def report_path(self) -> Optional[Path]:

        if self.report_directory:
            return Path(self.report_directory)

        return None

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def mark_completed(self) -> None:

        self.completed_at = datetime.now()

        self.execution_time = (
            self.completed_at - self.started_at
        ).total_seconds()

    def reset(self) -> None:

        self.dataset_name = ""

        self.dataset_version = ""

        self.dataset_path = ""

        self.report_directory = ""

        self.started_at = datetime.now()

        self.completed_at = None

        self.execution_time = 0.0

        self.user_metadata.clear()

    def to_dict(self) -> Dict[str, Any]:

        data = asdict(self)

        data["started_at"] = self.started_at.isoformat()

        data["completed_at"] = (
            self.completed_at.isoformat()
            if self.completed_at
            else None
        )

        return data
# =============================================================================
# Validation Issue
# =============================================================================

@dataclass(slots=True)
class ValidationIssue:
    """
    Represents a single validation issue detected during validation.
    """

    issue_id: str

    severity: ValidationSeverity

    category: ValidationCategory

    rule_name: str

    dataset: str = ""

    image_path: str = ""

    annotation_path: str = ""

    object_index: Optional[int] = None

    class_id: Optional[int] = None

    message: str = ""

    recommendation: str = ""

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    timestamp: datetime = field(
        default_factory=datetime.now
    )

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------

    @property
    def is_info(self) -> bool:
        return self.severity == ValidationSeverity.INFO

    @property
    def is_warning(self) -> bool:
        return self.severity == ValidationSeverity.WARNING

    @property
    def is_error(self) -> bool:
        return self.severity == ValidationSeverity.ERROR

    @property
    def is_critical(self) -> bool:
        return self.severity == ValidationSeverity.CRITICAL

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:

        return {
            "issue_id": self.issue_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "rule_name": self.rule_name,
            "dataset": self.dataset,
            "image_path": self.image_path,
            "annotation_path": self.annotation_path,
            "object_index": self.object_index,
            "class_id": self.class_id,
            "message": self.message,
            "recommendation": self.recommendation,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "ValidationIssue":

        return cls(
            issue_id=data["issue_id"],
            severity=ValidationSeverity(data["severity"]),
            category=ValidationCategory(data["category"]),
            rule_name=data["rule_name"],
            dataset=data.get("dataset", ""),
            image_path=data.get("image_path", ""),
            annotation_path=data.get("annotation_path", ""),
            object_index=data.get("object_index"),
            class_id=data.get("class_id"),
            message=data.get("message", ""),
            recommendation=data.get("recommendation", ""),
            details=data.get("details", {}),
            timestamp=datetime.fromisoformat(
                data["timestamp"]
            ) if "timestamp" in data else datetime.now(),
        )

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def __str__(self) -> str:

        return (
            f"[{self.severity.value}] "
            f"{self.rule_name}: "
            f"{self.message}"
        )

    def __repr__(self) -> str:

        return (
            "ValidationIssue("
            f"severity={self.severity.value}, "
            f"category={self.category.value}, "
            f"rule='{self.rule_name}'"
            ")"
        )
# =============================================================================
# Validation Summary
# =============================================================================

@dataclass(slots=True)
class ValidationSummary:
    """
    High-level summary of a validation run.
    """

    # -------------------------------------------------------------------------
    # Validation Status
    # -------------------------------------------------------------------------

    passed: bool = True

    # -------------------------------------------------------------------------
    # Dataset Summary
    # -------------------------------------------------------------------------

    total_images: int = 0

    total_objects: int = 0

    total_issues: int = 0

    # -------------------------------------------------------------------------
    # Issue Counts
    # -------------------------------------------------------------------------

    info: int = 0

    warnings: int = 0

    errors: int = 0

    critical: int = 0

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    validation_time: float = 0.0

    # -------------------------------------------------------------------------
    # Compatibility Aliases
    # -------------------------------------------------------------------------

    @property
    def information(self) -> int:
        """
        Compatibility alias for older code.
        """
        return self.info

    @information.setter
    def information(self, value: int):
        self.info = value

    # -------------------------------------------------------------------------
    # Derived Properties
    # -------------------------------------------------------------------------

    @property
    def failed(self) -> bool:
        return not self.passed

    @property
    def total_failures(self) -> int:
        return self.errors + self.critical

    @property
    def success_rate(self) -> float:

        if self.total_issues == 0:
            return 100.0

        successful = self.total_issues - self.total_failures

        return (successful / self.total_issues) * 100.0

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def reset(self) -> None:

        self.passed = True

        self.total_images = 0

        self.total_objects = 0

        self.total_issues = 0

        self.info = 0

        self.warnings = 0

        self.errors = 0

        self.critical = 0

        self.validation_time = 0.0

    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)

    def __repr__(self) -> str:

        return (
            "ValidationSummary("
            f"passed={self.passed}, "
            f"issues={self.total_issues}, "
            f"errors={self.errors}, "
            f"warnings={self.warnings}, "
            f"critical={self.critical}"
            ")"
        )
# =============================================================================
# Validation Statistics
# =============================================================================

@dataclass(slots=True)
class ValidationStatistics:
    """
    Dataset statistics computed after validation.
    """

    # -------------------------------------------------------------------------
    # Dataset Counts
    # -------------------------------------------------------------------------

    total_images: int = 0

    total_objects: int = 0

    total_bounding_boxes: int = 0

    # -------------------------------------------------------------------------
    # Image Dimensions
    # -------------------------------------------------------------------------

    minimum_width: float = 0.0

    maximum_width: float = 0.0

    average_width: float = 0.0

    minimum_height: float = 0.0

    maximum_height: float = 0.0

    average_height: float = 0.0

    # -------------------------------------------------------------------------
    # Objects Per Image
    # -------------------------------------------------------------------------

    objects_per_image: List[int] = field(
        default_factory=list
    )

    minimum_objects_per_image: int = 0

    maximum_objects_per_image: int = 0

    empty_images: int = 0

    # -------------------------------------------------------------------------
    # Bounding Box Statistics
    # -------------------------------------------------------------------------

    minimum_bbox_width: float = 0.0

    maximum_bbox_width: float = 0.0

    average_bbox_width: float = 0.0

    minimum_bbox_height: float = 0.0

    maximum_bbox_height: float = 0.0

    average_bbox_height: float = 0.0

    minimum_bbox_area: float = 0.0

    maximum_bbox_area: float = 0.0

    average_bbox_area: float = 0.0

    # -------------------------------------------------------------------------
    # Dataset Distribution
    # -------------------------------------------------------------------------

    dataset_counts: Dict[str, int] = field(
        default_factory=dict
    )

    class_counts: Dict[str, int] = field(
        default_factory=dict
    )

    split_counts: Dict[str, int] = field(
        default_factory=dict
    )

    image_formats: Dict[str, int] = field(
        default_factory=dict
    )

    # -------------------------------------------------------------------------
    # Compatibility Property
    # -------------------------------------------------------------------------

    @property
    def average_objects_per_image(self) -> float:

        if self.objects_per_image:
            return (
                sum(self.objects_per_image)
                / len(self.objects_per_image)
            )

        if self.total_images > 0:
            return (
                self.total_objects
                / self.total_images
            )

        return 0.0

    @average_objects_per_image.setter
    def average_objects_per_image(
        self,
        value: float,
    ):
        """
        Compatibility setter.

        validator.py assigns directly to this property.
        We intentionally ignore the assigned value because
        the average is computed from objects_per_image or
        total_objects/total_images.
        """
        pass

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def reset(self) -> None:

        self.total_images = 0

        self.total_objects = 0

        self.total_bounding_boxes = 0

        self.minimum_width = 0.0
        self.maximum_width = 0.0
        self.average_width = 0.0

        self.minimum_height = 0.0
        self.maximum_height = 0.0
        self.average_height = 0.0

        self.objects_per_image.clear()

        self.minimum_objects_per_image = 0
        self.maximum_objects_per_image = 0

        self.empty_images = 0

        self.minimum_bbox_width = 0.0
        self.maximum_bbox_width = 0.0
        self.average_bbox_width = 0.0

        self.minimum_bbox_height = 0.0
        self.maximum_bbox_height = 0.0
        self.average_bbox_height = 0.0

        self.minimum_bbox_area = 0.0
        self.maximum_bbox_area = 0.0
        self.average_bbox_area = 0.0

        self.dataset_counts.clear()

        self.class_counts.clear()

        self.split_counts.clear()

        self.image_formats.clear()

    def to_dict(self) -> Dict[str, Any]:

        return {
            "total_images": self.total_images,
            "total_objects": self.total_objects,
            "total_bounding_boxes": self.total_bounding_boxes,
            "minimum_width": self.minimum_width,
            "maximum_width": self.maximum_width,
            "average_width": self.average_width,
            "minimum_height": self.minimum_height,
            "maximum_height": self.maximum_height,
            "average_height": self.average_height,
            "minimum_objects_per_image": self.minimum_objects_per_image,
            "maximum_objects_per_image": self.maximum_objects_per_image,
            "average_objects_per_image": self.average_objects_per_image,
            "empty_images": self.empty_images,
            "minimum_bbox_width": self.minimum_bbox_width,
            "maximum_bbox_width": self.maximum_bbox_width,
            "average_bbox_width": self.average_bbox_width,
            "minimum_bbox_height": self.minimum_bbox_height,
            "maximum_bbox_height": self.maximum_bbox_height,
            "average_bbox_height": self.average_bbox_height,
            "minimum_bbox_area": self.minimum_bbox_area,
            "maximum_bbox_area": self.maximum_bbox_area,
            "average_bbox_area": self.average_bbox_area,
            "dataset_counts": self.dataset_counts,
            "class_counts": self.class_counts,
            "split_counts": self.split_counts,
            "image_formats": self.image_formats,
        }

    def __repr__(self) -> str:

        return (
            "ValidationStatistics("
            f"images={self.total_images}, "
            f"objects={self.total_objects}, "
            f"boxes={self.total_bounding_boxes}"
            ")"
        )
# =============================================================================
# Validation Metrics
# =============================================================================

@dataclass(slots=True)
class ValidationMetrics:
    """
    Quality metrics computed after validation.
    All scores range from 0 to 100.
    """

    # -------------------------------------------------------------------------
    # Core Quality Scores
    # -------------------------------------------------------------------------

    annotation_completeness_score: float = 100.0

    image_integrity_score: float = 100.0

    class_consistency_score: float = 100.0

    bounding_box_validity_score: float = 100.0

    blood_dataset_quality_index: float = 100.0

    overall_grade: str = "A+"

    # -------------------------------------------------------------------------
    # Compatibility Alias
    # -------------------------------------------------------------------------

    @property
    def bdqi_score(self) -> float:
        """
        Compatibility alias for older code.
        """
        return self.blood_dataset_quality_index

    @bdqi_score.setter
    def bdqi_score(self, value: float):
        self.blood_dataset_quality_index = value

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def reset(self) -> None:

        self.annotation_completeness_score = 100.0

        self.image_integrity_score = 100.0

        self.class_consistency_score = 100.0

        self.bounding_box_validity_score = 100.0

        self.blood_dataset_quality_index = 100.0

        self.overall_grade = "A+"

    def to_dict(self) -> Dict[str, Any]:

        return {
            "annotation_completeness_score":
                self.annotation_completeness_score,

            "image_integrity_score":
                self.image_integrity_score,

            "class_consistency_score":
                self.class_consistency_score,

            "bounding_box_validity_score":
                self.bounding_box_validity_score,

            "blood_dataset_quality_index":
                self.blood_dataset_quality_index,

            "overall_grade":
                self.overall_grade,
        }

    def __repr__(self) -> str:

        return (
            "ValidationMetrics("
            f"BDQI={self.blood_dataset_quality_index:.2f}, "
            f"Grade='{self.overall_grade}'"
            ")"
        )
# =============================================================================
# Validation Result
# =============================================================================

@dataclass
class ValidationResult:
    """
    Complete result produced by the validation engine.
    """

    metadata: ValidationMetadata = field(
        default_factory=ValidationMetadata
    )

    summary: ValidationSummary = field(
        default_factory=ValidationSummary
    )

    statistics: ValidationStatistics = field(
        default_factory=ValidationStatistics
    )

    metrics: ValidationMetrics = field(
        default_factory=ValidationMetrics
    )

    issues: List[ValidationIssue] = field(
        default_factory=list
    )

    # -------------------------------------------------------------------------
    # Issue Management
    # -------------------------------------------------------------------------

    def add_issue(self, issue: ValidationIssue) -> None:

        self.issues.append(issue)

        self.summary.total_issues += 1

        if issue.severity == ValidationSeverity.INFO:
            self.summary.info += 1

        elif issue.severity == ValidationSeverity.WARNING:
            self.summary.warnings += 1

        elif issue.severity == ValidationSeverity.ERROR:
            self.summary.errors += 1
            self.summary.passed = False

        elif issue.severity == ValidationSeverity.CRITICAL:
            self.summary.critical += 1
            self.summary.passed = False

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------

    @property
    def passed(self) -> bool:
        return self.summary.passed

    @property
    def failed(self) -> bool:
        return not self.summary.passed

    @property
    def error_count(self) -> int:
        return self.summary.errors

    @property
    def warning_count(self) -> int:
        return self.summary.warnings

    @property
    def critical_count(self) -> int:
        return self.summary.critical

    @property
    def info_count(self) -> int:
        return self.summary.info

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    def get_issues_by_severity(
        self,
        severity: ValidationSeverity,
    ) -> List[ValidationIssue]:

        return [
            issue
            for issue in self.issues
            if issue.severity == severity
        ]

    def get_issues_by_category(
        self,
        category: ValidationCategory,
    ) -> List[ValidationIssue]:

        return [
            issue
            for issue in self.issues
            if issue.category == category
        ]

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def reset(self) -> None:

        self.metadata.reset()

        self.summary.reset()

        self.statistics.reset()

        self.metrics.reset()

        self.issues.clear()

    def to_dict(self) -> Dict[str, Any]:

        return {
            "metadata": self.metadata.to_dict(),
            "summary": self.summary.to_dict(),
            "statistics": self.statistics.to_dict(),
            "metrics": self.metrics.to_dict(),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }

    def __len__(self) -> int:
        return len(self.issues)

    def __bool__(self) -> bool:
        return self.summary.passed

    def __repr__(self) -> str:

        return (
            "ValidationResult("
            f"passed={self.summary.passed}, "
            f"issues={len(self.issues)}, "
            f"grade='{self.metrics.overall_grade}'"
            ")"
        )
# =============================================================================
# Factory Functions
# =============================================================================

def create_validation_result() -> ValidationResult:
    """
    Create an empty ValidationResult.
    """
    return ValidationResult()


def create_validation_issue(
    issue_id: str,
    severity: ValidationSeverity,
    category: ValidationCategory,
    rule_name: str,
    message: str,
    recommendation: str = "",
    **kwargs,
) -> ValidationIssue:
    """
    Convenience factory for creating ValidationIssue objects.
    """

    return ValidationIssue(
        issue_id=issue_id,
        severity=severity,
        category=category,
        rule_name=rule_name,
        message=message,
        recommendation=recommendation,
        dataset=kwargs.get("dataset", ""),
        image_path=kwargs.get("image_path", ""),
        annotation_path=kwargs.get("annotation_path", ""),
        object_index=kwargs.get("object_index"),
        class_id=kwargs.get("class_id"),
        details=kwargs.get("details", {}),
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationMetadata",
    "ValidationIssue",
    "ValidationSummary",
    "ValidationStatistics",
    "ValidationMetrics",
    "ValidationResult",
    "create_validation_result",
    "create_validation_issue",
]