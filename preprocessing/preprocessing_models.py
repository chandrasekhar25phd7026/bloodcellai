"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    preprocessing_models.py

Version:
    1.0.0

Description
-----------
Core data models for the BloodCellAI preprocessing framework.

These models provide a unified representation for image metadata,
preprocessing operations, image quality assessment, execution
statistics, and preprocessing results.

Design Principles
-----------------
✓ Framework Independent
✓ Immutable Data Models (where appropriate)
✓ Enterprise Architecture
✓ Type Safe
✓ Extensible
✓ Research Reproducibility

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# Enumerations
# =============================================================================

class TransformStatus(str, Enum):
    """
    Status of a preprocessing transform.
    """

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class InterpolationMethod(str, Enum):
    """
    Supported interpolation methods for image resizing.
    """

    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    AREA = "area"
    LANCZOS = "lanczos"


class NormalizationMethod(str, Enum):
    """
    Supported image normalization methods.
    """

    NONE = "none"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    IMAGENET = "imagenet"
    UNIT_SCALE = "unit_scale"
    CUSTOM = "custom"


# =============================================================================
# Image Metadata
# =============================================================================

@dataclass(slots=True)
class ImageMetadata:
    """
    Stores metadata describing an image.

    This class contains only descriptive information and does not
    contain image pixel data.
    """

    image_path: Optional[Path] = None

    file_name: str = ""

    dataset_name: str = ""

    width: int = 0

    height: int = 0

    channels: int = 0

    dtype: str = ""

    color_space: str = "RGB"

    file_size: int = 0

    created_at: datetime = field(default_factory=datetime.now)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Return image size as (width, height).
        """
        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        """
        Compute image aspect ratio.

        Returns
        -------
        float
            Width / Height
        """

        if self.height == 0:
            return 0.0

        return self.width / self.height

    @property
    def total_pixels(self) -> int:
        """
        Total number of pixels.
        """

        return self.width * self.height

    @property
    def is_color(self) -> bool:
        """
        True if image has three or more channels.
        """

        return self.channels >= 3

    @property
    def is_grayscale(self) -> bool:
        """
        True if image has one channel.
        """

        return self.channels == 1

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metadata to a dictionary.
        """

        return {
            "image_path": str(self.image_path)
            if self.image_path else None,
            "file_name": self.file_name,
            "dataset_name": self.dataset_name,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "dtype": self.dtype,
            "color_space": self.color_space,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
# =============================================================================
# Transform Record
# =============================================================================

@dataclass(slots=True)
class TransformRecord:
    """
    Stores information about a single preprocessing transform.
    """

    name: str

    status: TransformStatus = TransformStatus.SUCCESS

    parameters: dict[str, Any] = field(default_factory=dict)

    execution_time: float = 0.0

    message: str = ""

    timestamp: datetime = field(default_factory=datetime.now)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        """
        True if the transform completed successfully.
        """
        return self.status == TransformStatus.SUCCESS

    @property
    def failed(self) -> bool:
        """
        True if the transform failed.
        """
        return self.status == TransformStatus.FAILED

    @property
    def skipped(self) -> bool:
        """
        True if the transform was skipped.
        """
        return self.status == TransformStatus.SKIPPED

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the transform record into a dictionary.
        """

        return {

            "name": self.name,

            "status": self.status.value,

            "parameters": self.parameters,

            "execution_time": self.execution_time,

            "message": self.message,

            "timestamp": self.timestamp.isoformat(),

            "metadata": self.metadata,

        }


# =============================================================================
# Image Quality Metrics
# =============================================================================

@dataclass(slots=True)
class ImageQualityMetrics:
    """
    Image quality measurements computed after preprocessing.
    """

    brightness: float = 0.0

    contrast: float = 0.0

    sharpness: float = 0.0

    blur_score: float = 0.0

    noise_level: float = 0.0

    entropy: float = 0.0

    dynamic_range: float = 0.0

    quality_score: float = 0.0

    # Settable, not a computed property: quality.py's
    # _evaluate_quality() explicitly assigns this based on whether
    # any threshold check produced a warning (a more nuanced signal
    # than a single quality_score cutoff) -- it used to be a
    # read-only @property (`quality_score >= 70.0`), which would
    # raise AttributeError the moment code tried to set it, as
    # QualityTransform does.
    passed: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:

        return {

            "brightness": self.brightness,

            "contrast": self.contrast,

            "sharpness": self.sharpness,

            "blur_score": self.blur_score,

            "noise_level": self.noise_level,

            "entropy": self.entropy,

            "dynamic_range": self.dynamic_range,

            "quality_score": self.quality_score,

            "metadata": self.metadata,

        }


# =============================================================================
# Preprocessing Statistics
# =============================================================================

@dataclass(slots=True)
class PreprocessingStatistics:
    """
    Runtime statistics for the preprocessing pipeline.
    """

    transforms_applied: int = 0

    transforms_skipped: int = 0

    transforms_failed: int = 0

    total_execution_time: float = 0.0

    warning_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_transforms(self) -> int:
        """
        Total number of transforms processed.
        """

        return (

            self.transforms_applied

            + self.transforms_skipped

            + self.transforms_failed

        )

    @property
    def success_rate(self) -> float:
        """
        Percentage of successful transforms.
        """

        if self.total_transforms == 0:
            return 0.0

        return (

            self.transforms_applied

            / self.total_transforms

        ) * 100.0

    def reset(self) -> None:
        """
        Reset all statistics.
        """

        self.transforms_applied = 0

        self.transforms_skipped = 0

        self.transforms_failed = 0

        self.total_execution_time = 0.0

        self.warning_count = 0

        self.metadata.clear()

    def to_dict(self) -> dict[str, Any]:

        return {

            "transforms_applied": self.transforms_applied,

            "transforms_skipped": self.transforms_skipped,

            "transforms_failed": self.transforms_failed,

            "total_transforms": self.total_transforms,

            "success_rate": self.success_rate,

            "total_execution_time": self.total_execution_time,

            "warning_count": self.warning_count,

            "metadata": self.metadata,

        }
# =============================================================================
# Preprocessing Result
# =============================================================================

@dataclass(slots=True)
class PreprocessingResult:
    """
    Represents the complete output of the preprocessing pipeline.
    """

    original_image: Any = None

    processed_image: Any = None

    metadata: ImageMetadata = field(
        default_factory=ImageMetadata
    )

    quality_metrics: ImageQualityMetrics = field(
        default_factory=ImageQualityMetrics
    )

    statistics: PreprocessingStatistics = field(
        default_factory=PreprocessingStatistics
    )

    transform_history: list[TransformRecord] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata_store: dict[str, Any] = field(
        default_factory=dict
    )

    passed: bool = True

    # ------------------------------------------------------------------
    # Transform Management
    # ------------------------------------------------------------------

    def add_transform(
        self,
        record: TransformRecord,
    ) -> None:
        """
        Add a preprocessing transform record.
        """

        self.transform_history.append(record)

        if record.successful:

            self.statistics.transforms_applied += 1

        elif record.skipped:

            self.statistics.transforms_skipped += 1

        else:

            self.statistics.transforms_failed += 1

            self.passed = False

        self.statistics.total_execution_time += (
            record.execution_time
        )

    # ------------------------------------------------------------------
    # Warning Management
    # ------------------------------------------------------------------

    def add_warning(
        self,
        warning: str,
    ) -> None:
        """
        Add a preprocessing warning.
        """

        if not warning:
            return

        self.warnings.append(warning)

        self.statistics.warning_count += 1

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Store additional metadata.
        """

        self.metadata_store[key] = value

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Retrieve metadata.
        """

        return self.metadata_store.get(
            key,
            default,
        )

    # ------------------------------------------------------------------
    # Helper Properties
    # ------------------------------------------------------------------

    @property
    def transform_count(self) -> int:
        """
        Total number of transforms executed.
        """

        return len(
            self.transform_history
        )

    @property
    def successful(self) -> bool:
        """
        Returns True if preprocessing completed successfully.
        """

        return self.passed

    @property
    def failed(self) -> bool:
        """
        Returns True if preprocessing failed.
        """

        return not self.passed

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset preprocessing result.
        """

        self.processed_image = None

        self.transform_history.clear()

        self.warnings.clear()

        self.metadata_store.clear()

        self.statistics.reset()

        self.quality_metrics = (
            ImageQualityMetrics()
        )

        self.passed = True

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Convert preprocessing result into a dictionary.
        """

        return {

            "metadata":
                self.metadata.to_dict(),

            "quality_metrics":
                self.quality_metrics.to_dict(),

            "statistics":
                self.statistics.to_dict(),

            "transform_history": [

                transform.to_dict()

                for transform

                in self.transform_history

            ],

            "warnings":

                self.warnings,

            "metadata_store":

                self.metadata_store,

            "passed":

                self.passed,

        }