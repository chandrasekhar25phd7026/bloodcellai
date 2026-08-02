"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    preprocessing_config.py

Version:
    1.0.0

Description
-----------
Central configuration models for the BloodCellAI preprocessing framework.

All preprocessing modules obtain their configuration from this file.
No preprocessing algorithm should use hard-coded parameters.

Design Principles
-----------------
✓ Single Source of Truth
✓ Type Safe
✓ Modular Configuration
✓ Framework Independent
✓ Extensible
✓ Enterprise Architecture

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


from .preprocessing_models import (
    InterpolationMethod,
    NormalizationMethod,
)


# =============================================================================
# Resize Configuration
# =============================================================================

@dataclass(slots=True)
class ResizeConfig:
    """
    Configuration for image resizing.
    """

    enabled: bool = True

    target_width: int = 640

    target_height: int = 640

    keep_aspect_ratio: bool = True

    pad_image: bool = True

    padding_value: int = 0

    interpolation: InterpolationMethod = (
        InterpolationMethod.LINEAR
    )

    def image_size(self) -> tuple[int, int]:
        """
        Return target image size.
        """

        return (
            self.target_width,
            self.target_height,
        )


# =============================================================================
# Normalization Configuration
# =============================================================================

@dataclass(slots=True)
class NormalizeConfig:
    """
    Configuration for image normalization.
    """

    enabled: bool = True

    method: NormalizationMethod = (
        NormalizationMethod.MIN_MAX
    )

    scale_min: float = 0.0

    scale_max: float = 1.0

    mean: tuple[float, float, float] = (
        0.485,
        0.456,
        0.406,
    )

    std: tuple[float, float, float] = (
        0.229,
        0.224,
        0.225,
    )

    clip_values: bool = True

    custom_parameters: dict = field(
        default_factory=dict
    )

    @property
    def use_imagenet(self) -> bool:
        """
        Returns True when ImageNet normalization
        is selected.
        """

        return (
            self.method
            == NormalizationMethod.IMAGENET
        )

    @property
    def use_minmax(self) -> bool:
        """
        Returns True when Min-Max normalization
        is selected.
        """

        return (
            self.method
            == NormalizationMethod.MIN_MAX
        )

    @property
    def use_zscore(self) -> bool:
        """
        Returns True when Z-score normalization
        is selected.
        """

        return (
            self.method
            == NormalizationMethod.Z_SCORE
        )
# =============================================================================
# CLAHE Configuration
# =============================================================================

@dataclass(slots=True)
class CLAHEConfig:
    """
    Configuration for Contrast Limited Adaptive Histogram Equalization.
    """

    enabled: bool = False

    clip_limit: float = 2.0

    tile_grid_size: tuple[int, int] = (8, 8)

    apply_to_luminance_only: bool = True

    color_space: str = "LAB"

    preserve_brightness: bool = True

    def tile_area(self) -> int:
        """
        Return tile area.
        """

        return (
            self.tile_grid_size[0]
            * self.tile_grid_size[1]
        )


# =============================================================================
# Denoising Configuration
# =============================================================================

@dataclass(slots=True)
class DenoiseConfig:
    """
    Configuration for image denoising.
    """

    enabled: bool = False

    method: str = "gaussian"

    kernel_size: int = 3

    sigma: float = 1.0

    strength: float = 10.0

    preserve_edges: bool = True

    custom_parameters: dict = field(
        default_factory=dict
    )

    @property
    def use_gaussian(self) -> bool:
        return self.method.lower() == "gaussian"

    @property
    def use_median(self) -> bool:
        return self.method.lower() == "median"

    @property
    def use_bilateral(self) -> bool:
        return self.method.lower() == "bilateral"

    @property
    def use_nlm(self) -> bool:
        return self.method.lower() in (
            "nlm",
            "non_local_means",
        )


# =============================================================================
# Sharpen Configuration
# =============================================================================

@dataclass(slots=True)
class SharpenConfig:
    """
    Configuration for image sharpening.
    """

    enabled: bool = False

    method: str = "unsharp"

    strength: float = 1.0

    kernel_size: int = 3

    sigma: float = 1.0

    amount: float = 1.5

    threshold: int = 0

    custom_parameters: dict = field(
        default_factory=dict
    )

    @property
    def use_unsharp(self) -> bool:
        return self.method.lower() == "unsharp"

    @property
    def use_kernel(self) -> bool:
        return self.method.lower() == "kernel"


# =============================================================================
# Color Configuration
# =============================================================================

@dataclass(slots=True)
class ColorConfig:
    """
    Configuration for color correction and enhancement.
    """

    enabled: bool = False

    auto_white_balance: bool = False

    gamma_correction: bool = False

    gamma: float = 1.0

    brightness: float = 0.0

    contrast: float = 1.0

    saturation: float = 1.0

    hue_shift: float = 0.0

    preserve_color_balance: bool = True

    custom_parameters: dict = field(
        default_factory=dict
    )

    @property
    def color_adjustment_enabled(self) -> bool:
        """
        Returns True if any color adjustment is enabled.
        """

        return any(

            (

                self.auto_white_balance,

                self.gamma_correction,

                self.brightness != 0.0,

                self.contrast != 1.0,

                self.saturation != 1.0,

                self.hue_shift != 0.0,

            )

        )
# =============================================================================
# Quality Configuration
# =============================================================================

@dataclass(slots=True)
class QualityConfig:
    """
    Configuration for image quality assessment.
    """

    enabled: bool = True

    minimum_brightness: float = 20.0

    maximum_brightness: float = 235.0

    minimum_contrast: float = 20.0

    minimum_sharpness: float = 50.0

    maximum_blur: float = 200.0

    minimum_entropy: float = 3.0

    minimum_quality_score: float = 70.0

    reject_low_quality_images: bool = False

    custom_thresholds: dict[str, float] = field(
        default_factory=dict
    )


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass(slots=True)
class PipelineConfig:
    """
    Configuration controlling preprocessing pipeline behavior.
    """

    enabled: bool = True

    save_intermediate_images: bool = False

    overwrite_existing: bool = True

    log_each_transform: bool = True

    continue_on_failure: bool = False

    raise_exceptions: bool = True

    compute_quality_metrics: bool = True

    record_transform_history: bool = True

    use_parallel_processing: bool = False

    number_of_workers: int = 1

    random_seed: int = 42


# =============================================================================
# Master Preprocessing Configuration
# =============================================================================

@dataclass(slots=True)
class PreprocessingConfig:
    """
    Master configuration for the BloodCellAI preprocessing framework.
    """

    resize: ResizeConfig = field(
        default_factory=ResizeConfig
    )

    normalize: NormalizeConfig = field(
        default_factory=NormalizeConfig
    )

    clahe: CLAHEConfig = field(
        default_factory=CLAHEConfig
    )

    denoise: DenoiseConfig = field(
        default_factory=DenoiseConfig
    )

    sharpen: SharpenConfig = field(
        default_factory=SharpenConfig
    )

    color: ColorConfig = field(
        default_factory=ColorConfig
    )

    quality: QualityConfig = field(
        default_factory=QualityConfig
    )

    pipeline: PipelineConfig = field(
        default_factory=PipelineConfig
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate preprocessing configuration.
        Raises ValueError if invalid.
        """

        if self.resize.target_width <= 0:
            raise ValueError(
                "Target width must be greater than zero."
            )

        if self.resize.target_height <= 0:
            raise ValueError(
                "Target height must be greater than zero."
            )

        if self.clahe.clip_limit <= 0:
            raise ValueError(
                "CLAHE clip_limit must be positive."
            )

        if self.denoise.kernel_size <= 0:
            raise ValueError(
                "Kernel size must be greater than zero."
            )

        if self.sharpen.kernel_size <= 0:
            raise ValueError(
                "Sharpen kernel size must be greater than zero."
            )

        if self.pipeline.number_of_workers < 1:
            raise ValueError(
                "Number of workers must be at least one."
            )

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Convert configuration into a dictionary.
        """

        return asdict(self)

    # -------------------------------------------------------------------------
    # Helper Properties
    # -------------------------------------------------------------------------

    @property
    def target_size(self) -> tuple[int, int]:
        """
        Return target resize dimensions.
        """

        return (
            self.resize.target_width,
            self.resize.target_height,
        )

    @property
    def preprocessing_enabled(self) -> bool:
        """
        Returns True if preprocessing pipeline is enabled.
        """

        return self.pipeline.enabled

    @property
    def enabled_steps(self) -> list[str]:
        """
        Return the list of enabled preprocessing steps.
        """

        steps = []

        if self.resize.enabled:
            steps.append("resize")

        if self.normalize.enabled:
            steps.append("normalize")

        if self.clahe.enabled:
            steps.append("clahe")

        if self.denoise.enabled:
            steps.append("denoise")

        if self.sharpen.enabled:
            steps.append("sharpen")

        if self.color.enabled:
            steps.append("color")

        return steps