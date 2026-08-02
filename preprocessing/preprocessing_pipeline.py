"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    preprocessing_pipeline.py

Description
-----------
Coordinates execution of all preprocessing transforms.

Responsibilities
----------------
✓ Build preprocessing pipeline
✓ Execute transforms
✓ Collect history
✓ Produce preprocessing results
✓ Handle failures

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from typing import List

import numpy as np
import logging
import time

from .clahe import CLAHETransform
from .color_balance import ColorBalanceTransform
from .denoise import DenoiseTransform
from .normalize import NormalizeTransform
from .preprocessing_config import PreprocessingConfig
from .quality import QualityTransform
from .resize import ResizeTransform
from .sharpen import SharpenTransform
from transforms.base_transform import BaseTransform
from transforms.compose import Compose

from .preprocessing_models import (
    ImageQualityMetrics,
    PreprocessingResult,
)


class PreprocessingPipeline:
    """
    Executes the complete BloodCellAI preprocessing pipeline.
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        self._config = config

        self._logger = logging.getLogger(
            self.__class__.__name__
        )

        self._pipeline = self._build_pipeline()

        self._last_result: PreprocessingResult | None = None

        self._execution_time = 0.0

    # -------------------------------------------------------------------------
    # Build Pipeline
    # -------------------------------------------------------------------------

    def _build_pipeline(
        self,
    ) -> Compose:
        """
        Build the preprocessing pipeline.

        NOTE: this used to be missing its `return` statement and most
        of its transform appends (sharpen/color/normalize/quality),
        which had been physically displaced to the very end of the
        file, stranded after unrelated methods -- meaning
        self._pipeline was silently set to None in __init__, and
        every use of the pipeline (transform_count, preprocess(),
        etc.) would have raised AttributeError on first use. Fixed by
        reassembling the full, correct transform order here.
        """

        transforms: List[BaseTransform] = []

        # Quality assessment runs FIRST, on the natural image, before
        # any transform below changes its pixel value range/dtype.
        # Brightness/contrast/blur/entropy are meaningful measurements
        # of the original image; running this after normalize (which
        # can produce a float32 tensor in [0,1] or z-scored range)
        # doesn't reflect the actual image's visual quality, and in
        # practice also hit real OpenCV dtype errors (cv2.Laplacian
        # and cv2.Sobel didn't support a float32->float64 combination
        # in this environment) -- confirmed by running this pipeline
        # against a real image.
        if self._config.quality.enabled:
            transforms.append(
                QualityTransform(self._config)
            )

        if self._config.resize.enabled:
            transforms.append(
                ResizeTransform(self._config)
            )

        if self._config.clahe.enabled:
            transforms.append(
                CLAHETransform(self._config)
            )

        if self._config.denoise.enabled:
            transforms.append(
                DenoiseTransform(self._config)
            )

        if self._config.sharpen.enabled:
            transforms.append(
                SharpenTransform(self._config)
            )

        if self._config.color.enabled:
            transforms.append(
                ColorBalanceTransform(self._config)
            )

        if self._config.normalize.enabled:
            transforms.append(
                NormalizeTransform(self._config)
            )

        return Compose(
            transforms
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def pipeline(
        self,
    ) -> Compose:
        """
        Return the composed preprocessing pipeline.
        """

        return self._pipeline

    @property
    def transform_count(
        self,
    ) -> int:
        """
        Return the number of enabled transforms.
        """

        return self._pipeline.count

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def preprocess(
        self,
        image: np.ndarray,
    ) -> PreprocessingResult:
        """
        Execute the preprocessing pipeline.

        NOTE: an earlier, incomplete duplicate of this method (missing
        validate(), timing, and exception handling) used to sit here
        too -- Python keeps only the last definition of a same-named
        method in a class, so that stub never actually ran; removed
        for clarity, keeping only this complete version.
        """

        self.validate()

        start_time = time.perf_counter()

        try:

            result = self._run_pipeline(image)

            self._last_result = result

            return result

        except Exception as error:

            self._logger.exception(
                "Pipeline execution failed."
            )

            raise error

        finally:

            self._execution_time = (
                time.perf_counter()
                - start_time
            )

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    def _run_pipeline(
        self,
        image: np.ndarray,
    ) -> PreprocessingResult:
        """
        Execute all enabled transforms.
        """

        return self._execute_transform(image)

    # -------------------------------------------------------------------------
    # Transform Execution
    # -------------------------------------------------------------------------

    def _execute_transform(
        self,
        image: np.ndarray,
    ) -> PreprocessingResult:
        """
        Execute the composed preprocessing pipeline.
        """

        return self._pipeline(image)

    # -------------------------------------------------------------------------
    # Latest Result
    # -------------------------------------------------------------------------

    @property
    def last_result(
        self,
    ) -> PreprocessingResult | None:
        """
        Return the most recent preprocessing result.
        """

        return self._last_result

    @property
    def processed_image(
        self,
    ) -> np.ndarray | None:
        """
        Return the latest processed image.

        NOTE: this used to reference `self._last_result.image`, but
        PreprocessingResult has no `.image` attribute at all (it's
        `.processed_image` / `.original_image`) -- this would have
        raised AttributeError on first real use.
        """

        if self._last_result is None:
            return None

        return self._last_result.processed_image

    @property
    def quality_metrics(
        self,
    ) -> ImageQualityMetrics | None:
        """
        Return quality metrics from the latest run.
        """

        if self._last_result is None:
            return None

        return self._last_result.quality_metrics

    @property
    def transform_history(
        self,
    ) -> list:
        """
        Return transform execution history.
        """

        if self._last_result is None:
            return []

        return self._last_result.transform_history

    @property
    def summary(
        self,
    ) -> dict:
        """
        Return a summary of the latest pipeline execution.
        """

        if self._last_result is None:

            return {
                "executed": False,
            }

        quality = self.quality_metrics

        return {

            "executed": True,

            "transform_count":
                self.transform_count,

            "successful_transforms":
                len(self.transform_history),

            "quality_score":
                (
                    quality.quality_score
                    if quality is not None
                    else None
                ),

            "passed_quality":
                (
                    quality.passed
                    if quality is not None
                    else None
                ),

            "warnings":
                (
                    quality.metadata.get(
                        "warnings",
                        [],
                    )
                    if quality is not None
                    else []
                ),
        }

    # -------------------------------------------------------------------------
    # Pipeline Management
    # -------------------------------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Reset the pipeline state.
        """

        self._last_result = None

        for transform in self._pipeline:

            transform.reset()

    def rebuild(
        self,
    ) -> None:
        """
        Rebuild the preprocessing pipeline.

        Useful after changing configuration values.
        """

        self.reset()

        self._pipeline = self._build_pipeline()

    @property
    def enabled_transforms(
        self,
    ) -> list[str]:
        """
        Return the names of all enabled transforms.
        """

        return [
            transform.__class__.__name__
            for transform in self._pipeline
        ]

    def pipeline_report(
        self,
    ) -> dict:
        """
        Generate a detailed pipeline report.

        NOTE: an earlier, less complete duplicate of this method (no
        execution_time) used to sit here too; removed, keeping only
        this version.
        """

        report = {

            "pipeline":
                self.__class__.__name__,

            "transform_count":
                self.transform_count,

            "enabled_transforms":
                self.enabled_transforms,

            "execution_time_seconds":
                round(
                    self.execution_time,
                    4,
                ),

            "executed":
                self._last_result is not None,
        }

        if self._last_result is not None:

            report["summary"] = self.summary

        return report

    def __len__(
        self,
    ) -> int:
        """
        Return the number of enabled transforms.
        """

        return self.transform_count

    def __iter__(
        self,
    ):
        """
        Iterate over enabled transforms.
        """

        return iter(
            self._pipeline
        )

    def __repr__(
        self,
    ) -> str:
        """
        Return a developer-friendly representation.
        """

        return (
            f"{self.__class__.__name__}"
            f"(transforms={self.transform_count})"
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(
        self,
    ) -> None:
        """
        Validate the preprocessing pipeline.
        """

        if self.transform_count == 0:
            raise ValueError(
                "Pipeline contains no enabled transforms."
            )

    @property
    def execution_time(
        self,
    ) -> float:
        """
        Return the execution time (seconds)
        of the latest pipeline execution.
        """

        return self._execution_time

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    def preprocess_batch(
        self,
        images: list[np.ndarray],
    ) -> list[PreprocessingResult]:
        """
        Preprocess a batch of images.
        """

        results = []

        for image in images:

            results.append(
                self.preprocess(image)
            )

        return results

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def enable_logging(
        self,
        level: int = logging.INFO,
    ) -> None:
        """
        Enable pipeline logging.
        """

        logging.basicConfig(
            level=level,
        )

    def disable_logging(
        self,
    ) -> None:
        """
        Disable pipeline logging.
        """

        logging.disable(
            logging.CRITICAL,
        )
