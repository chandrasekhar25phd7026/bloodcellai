"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    preprocessing_manager.py

Description
-----------
Dataset-level preprocessing manager.

Responsibilities
----------------
✓ Process individual images
✓ Process directories
✓ Process datasets
✓ Save processed images
✓ Generate reports
✓ Collect statistics

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import time
import json
from datetime import datetime
import numpy as np

from .preprocessing_config import PreprocessingConfig
from .preprocessing_models import PreprocessingResult
from .preprocessing_pipeline import PreprocessingPipeline


class PreprocessingManager:
    """
    Dataset-level preprocessing manager.
    """
    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        self._config = config

        self._logger = logging.getLogger(
            self.__class__.__name__
        )

        self._pipeline = PreprocessingPipeline(
            config
        )

        self._processed_images = 0
        self._failed_images = 0

        self._results: list[
            PreprocessingResult
        ] = []
        self._processed_files: set[Path] = set()

        self._stop_on_error = False

        self._progress_callback = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def pipeline(
        self,
    ) -> PreprocessingPipeline:
        """
        Return the preprocessing pipeline.
        """

        return self._pipeline

    @property
    def processed_images(
        self,
    ) -> int:
        """
        Return the number of successfully processed images.
        """

        return self._processed_images

    @property
    def failed_images(
        self,
    ) -> int:
        """
        Return the number of failed images.
        """

        return self._failed_images

    @property
    def results(
        self,
    ) -> list[PreprocessingResult]:
        """
        Return all preprocessing results.
        """

        return self._results
    # -------------------------------------------------------------------------
    # Single Image Processing
    # -------------------------------------------------------------------------

    def preprocess_image(
        self,
        image: np.ndarray | str | Path,
    ) -> PreprocessingResult:
        """
        Preprocess a single image.

        Parameters
        ----------
        image
            NumPy image array or image file path.

        Returns
        -------
        PreprocessingResult
            Preprocessing result.
        """

        start_time = time.perf_counter()

        try:

            image_array = self._load_image(image)

            result = self._pipeline.preprocess(
                image_array
            )

            self._results.append(
                result
            )

            self._processed_images += 1

            return result

        except Exception:

            self._failed_images += 1

            self._logger.exception(
                "Failed to preprocess image."
            )

            raise

        finally:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            self._logger.debug(
                "Processing time: %.4f seconds",
                elapsed,
            )
    # -------------------------------------------------------------------------
    # Image Loading
    # -------------------------------------------------------------------------

    def _load_image(
        self,
        image: np.ndarray | str | Path,
    ) -> np.ndarray:
        """
        Load an image from memory or disk.
        """

        if isinstance(
            image,
            np.ndarray,
        ):
            return image

        image_path = Path(image)

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        loaded = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if loaded is None:

            raise ValueError(
                f"Unable to load image: {image_path}"
            )

        return loaded
    # -------------------------------------------------------------------------
    # Multiple Images
    # -------------------------------------------------------------------------

    def preprocess_images(
        self,
        images: list[
            np.ndarray | str | Path
        ],
    ) -> list[PreprocessingResult]:
        """
        Preprocess multiple images.
        """

        results = []

        for image in images:

            results.append(
                self.preprocess_image(
                    image
                )
            )

        return results
    # -------------------------------------------------------------------------
    # Directory Processing
    # -------------------------------------------------------------------------

    def preprocess_directory(
        self,
        input_directory: str | Path,
        output_directory: str | Path | None = None,
        recursive: bool = True,
        save_images: bool = False,
    ) -> list[PreprocessingResult]:
        """
        Preprocess every supported image inside a directory.

        Parameters
        ----------
        input_directory
            Directory containing images.

        output_directory
            Destination directory.

        recursive
            Search subdirectories recursively.

        save_images
            Save processed images.
        """

        input_directory = Path(
            input_directory
        )

        if not input_directory.exists():

            raise FileNotFoundError(
                input_directory
            )

        if not input_directory.is_dir():

            raise ValueError(
                "Input must be a directory."
            )

        image_paths = self._find_images(
            input_directory,
            recursive,
        )

        results = []

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):
            self._notify_progress(

                index,

                len(image_paths),

                image_path,

            )

            try:

                result = self.preprocess_image(
                    image_path
                )
                self._processed_files.add(
                    image_path
                )

                results.append(
                    result
                )

                if save_images:

                    self.save_image(
                        result,
                        image_path,
                        input_directory,
                        output_directory,
                    )

            except Exception as error:

                self._logger.exception(

                    "Failed processing %s",

                    image_path,
                )
                if self._stop_on_error:
                    raise error

                continue

        return results

    # -------------------------------------------------------------------------
    # Dataset Validation
    # -------------------------------------------------------------------------

    def validate_directory(
        self,
        directory: str | Path,
    ) -> None:
        """
        Validate an input dataset directory.
        """

        directory = Path(directory)

        if not directory.exists():

            raise FileNotFoundError(
                directory
            )

        if not directory.is_dir():

            raise ValueError(
                "Expected a directory."
            )

        images = self._find_images(
            directory,
            recursive=True,
        )

        if len(images) == 0:

            raise ValueError(
                "No supported images found."
            )
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def enable_file_logging(
        self,
        log_file: str | Path,
    ) -> None:
        """
        Enable logging to a file.
        """

        handler = logging.FileHandler(
            log_file,
        )

        formatter = logging.Formatter(

            "%(asctime)s "

            "%(levelname)s "

            "%(message)s"

        )

        handler.setFormatter(
            formatter
        )

        self._logger.addHandler(
            handler
        )

        self._logger.setLevel(
            logging.INFO
        )
    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Reset manager state.
        """

        self._processed_images = 0

        self._failed_images = 0

        self._results.clear()

        self._processed_files.clear()

        self._pipeline.reset()

    # -------------------------------------------------------------------------
    # Image Discovery
    # -------------------------------------------------------------------------

    def _find_images(
        self,
        directory: Path,
        recursive: bool,
    ) -> list[Path]:
        """
        Locate all supported images.
        """

        images = []

        if recursive:

            iterator = directory.rglob("*")

        else:

            iterator = directory.glob("*")

        for file in iterator:

            if (
                file.is_file()
                and file.suffix.lower()
                in self.SUPPORTED_EXTENSIONS
            ):

                images.append(file)

        images.sort()

        return images
    @property
    def total_images(
        self,
    ) -> int:
        """
        Total processed attempts.
        """

        return (

            self.processed_images

            +

            self.failed_images

        )
    @property
    def success_rate(
        self,
    ) -> float:
        """
        Processing success percentage.
        """

        if self.total_images == 0:

            return 0.0

        return (

            self.processed_images

            /

            self.total_images

        ) * 100.0

    # -------------------------------------------------------------------------
    # Save Processed Image
    # -------------------------------------------------------------------------

    def save_image(
        self,
        result: PreprocessingResult,
        source_path: Path,
        input_directory: Path,
        output_directory: str | Path | None,
    ) -> Path:
        """
        Save a processed image while preserving the directory structure.
        """

        if output_directory is None:

            raise ValueError(
                "Output directory cannot be None."
            )

        output_directory = Path(
            output_directory
        )

        output_path = self._build_output_path(
            source_path,
            input_directory,
            output_directory,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            result.image,
        )

        if not success:

            raise IOError(
                f"Unable to save image: {output_path}"
            )

        self.save_metadata(
            result,
            output_path,
        )

        return output_path
    # -------------------------------------------------------------------------
    # Output Path
    # -------------------------------------------------------------------------

    def _build_output_path(
        self,
        source_path: Path,
        input_directory: Path,
        output_directory: Path,
    ) -> Path:
        """
        Build the destination path while preserving
        the input directory structure.
        """

        relative_path = source_path.relative_to(
            input_directory
        )

        return output_directory / relative_path
    # -------------------------------------------------------------------------
    # Save Metadata
    # -------------------------------------------------------------------------

    def save_metadata(
        self,
        result: PreprocessingResult,
        image_path: Path,
    ) -> Path:
        """
        Save preprocessing metadata beside the image.
        """

        metadata_path = image_path.with_suffix(
            ".json"
        )

        metadata = self._metadata_to_dict(
            result
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(

                metadata,

                file,

                indent=4,

                ensure_ascii=False,

            )

        return metadata_path
    # -------------------------------------------------------------------------
    # Metadata Conversion
    # -------------------------------------------------------------------------

    def _metadata_to_dict(
        self,
        result: PreprocessingResult,
    ) -> dict:
        """
        Convert preprocessing results into a JSON-
        serializable dictionary.
        """

        metadata = {

            "transform_history": [],

            "statistics": {},

            "quality_metrics": {},

            "warnings": [],

        }

        if hasattr(
            result,
            "transform_history",
        ):

            metadata[
                "transform_history"
            ] = [

                record.to_dict()

                if hasattr(
                    record,
                    "to_dict",
                )

                else str(record)

                for record in result.transform_history

            ]

        if hasattr(
            result,
            "statistics",
        ):

            statistics = result.statistics

            if hasattr(
                statistics,
                "to_dict",
            ):

                metadata[
                    "statistics"
                ] = statistics.to_dict()

        if hasattr(
            result,
            "quality_metrics",
        ):

            quality = result.quality_metrics

            if hasattr(
                quality,
                "to_dict",
            ):

                metadata[
                    "quality_metrics"
                ] = quality.to_dict()

                metadata[
                    "warnings"
                ] = quality.metadata.get(
                    "warnings",
                    [],
                )

        return metadata
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    @property
    def statistics(
        self,
    ) -> dict:
        """
        Return preprocessing statistics.
        """

        accepted = 0
        rejected = 0

        quality_scores = []

        for result in self._results:

            quality = getattr(
                result,
                "quality_metrics",
                None,
            )

            if quality is None:
                continue

            quality_scores.append(
                quality.quality_score
            )

            if quality.passed:

                accepted += 1

            else:

                rejected += 1

        average_quality = (
            sum(quality_scores)
            / len(quality_scores)
            if quality_scores
            else 0.0
        )

        return {

            "processed_images":
                self.processed_images,

            "failed_images":
                self.failed_images,

            "total_images":
                self.total_images,

            "success_rate":
                round(
                    self.success_rate,
                    2,
                ),

            "accepted_images":
                accepted,

            "rejected_images":
                rejected,

            "average_quality_score":
                round(
                    average_quality,
                    2,
                ),

            "pipeline_execution_time":
                round(
                    self.pipeline.execution_time,
                    4,
                ),
        }
    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------

    def generate_report(
        self,
    ) -> dict:
        """
        Generate a complete preprocessing report.
        """

        report = {

            "framework":

                "BloodCellAI",

            "generated":

                datetime.now().isoformat(),

            "pipeline":

                self.pipeline.pipeline_report(),

            "statistics":

                self.statistics,

            "images":

                [],
        }

        for result in self.results:

            image_report = {

                "quality_score": None,

                "passed": None,

                "warnings": [],

            }

            quality = getattr(
                result,
                "quality_metrics",
                None,
            )

            if quality is not None:

                image_report[
                    "quality_score"
                ] = quality.quality_score

                image_report[
                    "passed"
                ] = quality.passed

                image_report[
                    "warnings"
                ] = quality.metadata.get(
                    "warnings",
                    [],
                )

            report["images"].append(
                image_report
            )

        return report
    # -------------------------------------------------------------------------
    # Export Report
    # -------------------------------------------------------------------------

    def export_report(
        self,
        output_file: str | Path,
    ) -> Path:
        """
        Export preprocessing report as JSON.
        """

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = self.generate_report()

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(

                report,

                file,

                indent=4,

                ensure_ascii=False,

            )

        return output_file
    # -------------------------------------------------------------------------
    # Console Summary
    # -------------------------------------------------------------------------

    def print_summary(
        self,
    ) -> None:
        """
        Print preprocessing summary.
        """

        stats = self.statistics

        print()

        print("=" * 60)

        print("BloodCellAI Preprocessing Summary")

        print("=" * 60)

        print(
            f"Processed Images : {stats['processed_images']}"
        )

        print(
            f"Failed Images    : {stats['failed_images']}"
        )

        print(
            f"Success Rate     : {stats['success_rate']:.2f}%"
        )

        print(
            f"Accepted Images  : {stats['accepted_images']}"
        )

        print(
            f"Rejected Images  : {stats['rejected_images']}"
        )

        print(
            "Average Quality  : "
            f"{stats['average_quality_score']:.2f}"
        )

        print(
            "Pipeline Time    : "
            f"{stats['pipeline_execution_time']:.4f}s"
        )

        print("=" * 60)

    # -------------------------------------------------------------------------
    # Progress Callback
    # -------------------------------------------------------------------------

    def set_progress_callback(
        self,
        callback,
    ) -> None:
        """
        Register a callback for processing progress.

        Signature:
            callback(current, total, image_path)
        """

        self._progress_callback = callback

    def _notify_progress(
        self,
        current: int,
        total: int,
        image_path: Path,
    ) -> None:

        if self._progress_callback is not None:

            self._progress_callback(
                current,
                total,
                image_path,
            )
