"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    validator.py

Version:
    4.0.0

Purpose:
    Enterprise Dataset Validation Engine

Description
-----------
Coordinates validation of a UniversalDataset by executing dataset,
image, object, and bounding-box validation rules, followed by
statistics and quality metric computation.

Design Principles
-----------------
✓ Single Responsibility
✓ Modular Rule Execution
✓ Enterprise Logging
✓ No Duplicate Validation Logic
✓ Compatible with BloodCellAI Models v3+

Note on imports:
    This module depends on UniversalDataset / UniversalImage /
    BoundingBox from the `bloodcell` package. The original version of
    this file imported them via `from ..builder.universal_dataset
    import ...`, which assumes this `validation/` folder is a
    subpackage of some root alongside a sibling `builder/` package.
    That layout does not match the actual bloodcell project (a flat
    package: `bloodcell/universal_dataset.py`, no `builder/`
    subpackage), so those imports would fail immediately. Fixed to
    import directly from `bloodcell`, which assumes this `validation/`
    package is placed as a sibling of (not nested inside) the
    `bloodcell/` package, with both importable from the same project
    root. If you instead nest this folder inside bloodcell/ itself,
    change these two imports to relative imports
    (`from ..universal_dataset import ...`).

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import Callable, Iterable

from .models import (
    ValidationResult,
    ValidationIssue,
    ValidationSummary,
    ValidationStatistics,
    ValidationMetrics,
)

from .statistics import compute_statistics
from .metrics import compute_metrics

from . import dataset_rules
from . import image_rules
from . import object_rules
from . import bbox_rules

from bloodcell.universal_dataset import UniversalDataset
from bloodcell.universal_object import (
    UniversalImage,
    BoundingBox,
)


# =============================================================================
# Logger
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Validator V2
# =============================================================================

class DatasetValidatorV2:
    """
    Enterprise Dataset Validator.

    Validation Pipeline
    -------------------

    1. Dataset Rules
    2. Image Rules
    3. Object Rules
    4. Bounding Box Rules
    5. Statistics
    6. Metrics

    This class only orchestrates validation.
    Validation logic resides entirely inside rule modules.
    """

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------

    def __init__(
        self,
        *,
        stop_on_error: bool = False,
        validate_images: bool = True,
        validate_objects: bool = True,
        validate_bboxes: bool = True,
        dataset_path: str | None = None,
        min_image_width: int = 1,
        min_image_height: int = 1,
        max_image_width: int = 100_000,
        max_image_height: int = 100_000,
        valid_class_ids: list[int] | None = None,
        expected_classes: list[str] | None = None,
        iou_threshold: float = 0.9,
        min_bbox_size: float = 0.0,
        max_bbox_aspect_ratio: float = 10.0,
    ) -> None:

        self.stop_on_error = stop_on_error

        self.validate_images = validate_images
        self.validate_objects = validate_objects
        self.validate_bboxes = validate_bboxes

        # `dataset_path`, if given, enables the dataset_rules.py checks
        # that need a real filesystem path (does the folder exist, are
        # there raw image/annotation files, etc.) -- these are
        # pre-build/filesystem-level checks and are gracefully skipped
        # (not treated as failures) when no path is available, since a
        # UniversalDataset built purely in memory may not have one.
        self.dataset_path = dataset_path

        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height

        # Optional, dataset-specific configuration for object_rules.py
        # checks that need it (check_invalid_class_id,
        # check_missing_classes). Left as None by default so those two
        # specific rules are skipped rather than run against a
        # meaningless default.
        self.valid_class_ids = valid_class_ids
        self.expected_classes = expected_classes

        self.iou_threshold = iou_threshold
        self.min_bbox_size = min_bbox_size
        self.max_bbox_aspect_ratio = max_bbox_aspect_ratio

        # Tracks image paths seen so far across the whole validation
        # run, for check_duplicate_image_path in image_rules.py -- this
        # needs to persist across images, not reset per-image.
        self._seen_image_paths: set[str] = set()

        self.result = ValidationResult()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _reset(self) -> None:
        """
        Reset validator state before starting a new validation.
        """

        self.result = ValidationResult()
        self._seen_image_paths = set()

    def _initialize(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Initialize validation metadata and summary.
        """

        summary = self.result.summary

        summary.total_images = len(dataset.images)

        summary.total_objects = sum(
            len(image.objects)
            for image in dataset.images
        )

        self.result.metadata.started_at = datetime.now()

    # -------------------------------------------------------------------------
    # Iterators
    # -------------------------------------------------------------------------

    def _iter_images(
        self,
        dataset: UniversalDataset,
    ) -> Iterable[UniversalImage]:
        """
        Iterate over every image in the dataset.
        """

        yield from dataset.images

    def _iter_objects(
        self,
        image: UniversalImage,
    ) -> Iterable[BoundingBox]:
        """
        Iterate over every object in an image.
        """

        yield from image.objects

    # -------------------------------------------------------------------------
    # Rule Discovery
    # -------------------------------------------------------------------------

    @staticmethod
    def _discover_rules(
        module,
        prefix: str = "check_",
    ) -> list[Callable]:
        """
        Automatically discover validation rule functions.

        Parameters
        ----------
        module
            Rule module.

        prefix
            Rule function prefix.

        Returns
        -------
        list[Callable]
            Sorted validation rule functions.
        """

        rules = []

        for name, obj in inspect.getmembers(
            module,
            inspect.isfunction,
        ):

            if not name.startswith(prefix):
                continue

            rules.append(obj)

        rules.sort(
            key=lambda rule: rule.__name__,
        )

        return rules

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    # NOTE: the full validate() implementation lives further down in
    # this class. An earlier, incomplete stub used to be defined here
    # too (same method name) -- Python silently keeps only the last
    # definition of a method in a class body, so that stub never
    # actually ran; it's removed here to avoid the confusion of two
    # definitions of the same method sitting in one class.

    # -------------------------------------------------------------------------
    # Issue Management
    # -------------------------------------------------------------------------

    def _add_issue(
        self,
        issue: ValidationIssue,
    ) -> None:
        """
        Add a ValidationIssue to the current ValidationResult.

        The ValidationResult is responsible for updating all
        summary counters and validation status.
        """

        if issue is None:
            return

        self.result.add_issue(issue)

    def _create_exception_issue(
        self,
        *,
        category,
        rule_name: str,
        dataset: str = "",
        image_path: str = "",
        message: str = "",
    ) -> ValidationIssue:
        """
        Create a ValidationIssue representing an internal
        validator exception.
        """

        from uuid import uuid4

        from .models import (
            ValidationSeverity,
            ValidationCategory,
        )

        if isinstance(category, str):

            category = {
                "dataset": ValidationCategory.DATASET,
                "image": ValidationCategory.IMAGE,
                "object": ValidationCategory.OBJECT,
                "bounding_box": ValidationCategory.BOUNDING_BOX,
                "bbox": ValidationCategory.BOUNDING_BOX,
            }.get(
                category.lower(),
                ValidationCategory.UNKNOWN,
            )

        return ValidationIssue(

            issue_id=str(uuid4()),

            severity=ValidationSeverity.ERROR,

            category=category,

            rule_name=rule_name,

            dataset=dataset,

            image_path=image_path,

            message=message,

            recommendation=(
                "Inspect the validation rule that raised "
                "the exception."
            ),
        )

    # -------------------------------------------------------------------------
    # Generic Rule Executor
    # -------------------------------------------------------------------------

    def _execute_rules(
        self,
        *,
        module,
        category,
        context: dict,
    ) -> None:
        """
        Execute every validation rule contained inside a rule module.

        Parameters
        ----------
        module
            Rule module.

        category
            ValidationCategory corresponding to the rule module, used
            only if a rule raises and we need to record an internal
            exception issue.

        context
            A dict of all values that might be needed by any rule in
            this module (e.g. {"image_path": ..., "dataset": ...,
            "result": ..., "objects": ..., "seen_paths": ...}). Each
            rule receives only the subset of these matching its own
            parameter names -- rules in the same module can (and do)
            declare different required/optional parameters.

        Design note
        -----------
        Every check_* function in dataset_rules.py / image_rules.py /
        object_rules.py / bbox_rules.py adds issues directly to the
        `result` object passed into it (a side effect) and returns
        None -- none of them return ValidationIssue objects. An
        earlier version of this method expected rules to return
        issues and process the return value; that code path was dead
        (every real rule returns None), so it's removed here rather
        than kept alongside working code.

        A rule is skipped (not treated as a failure) if the context
        doesn't contain a value for one of its *required* (no-default)
        parameters -- e.g. dataset_rules.py's filesystem checks need
        `dataset_path`, which isn't always available for an in-memory
        UniversalDataset. Skipped rules are logged at debug level only.
        """

        rules = self._discover_rules(module)

        logger.info(
            "Executing %d rule(s) from %s",
            len(rules),
            module.__name__,
        )

        for rule in rules:

            signature = inspect.signature(rule)

            missing_required = [
                name
                for name, param in signature.parameters.items()
                if name not in context
                and param.default is inspect.Parameter.empty
            ]

            if missing_required:

                logger.debug(
                    "Skipping %s: requires %s, not available in "
                    "this validation context.",
                    rule.__name__,
                    missing_required,
                )

                continue

            kwargs = {
                name: context[name]
                for name in signature.parameters
                if name in context
            }

            try:

                rule(**kwargs)

            except Exception as exc:

                logger.exception(
                    "Rule %s failed.",
                    rule.__name__,
                )

                issue = self._create_exception_issue(

                    category=category,

                    rule_name=rule.__name__,

                    dataset=str(
                        context.get(
                            "dataset",
                            context.get("dataset_name", ""),
                        )
                    ),

                    image_path=str(context.get("image_path", "")),

                    message=str(exc),

                )

                self._add_issue(issue)

                if self.stop_on_error:
                    return

                continue

        logger.info(
            "Finished executing %s",
            module.__name__,
        )

    # -------------------------------------------------------------------------
    # Object -> Dict Conversion
    # -------------------------------------------------------------------------

    @staticmethod
    def _bbox_to_dict(bbox: BoundingBox) -> dict:
        """
        Convert a BoundingBox dataclass instance into the plain dict
        shape object_rules.py / bbox_rules.py actually expect (they
        use dict access like `obj.get("bbox")` / `"class_id" not in
        obj`, which raises on a dataclass instance directly).

        The "bbox" key holds [xc, yc, w, h] -- the normalized
        center/width/height form BoundingBox already stores, matching
        bbox_rules.py's "yolo" bbox_format exactly.
        """

        return {
            "class_id": getattr(bbox, "class_id", None),
            "class_name": getattr(bbox, "class_name", None),
            "bbox": [
                getattr(bbox, "xc", None),
                getattr(bbox, "yc", None),
                getattr(bbox, "w", None),
                getattr(bbox, "h", None),
            ],
            "confidence": getattr(bbox, "confidence", None),
        }

    @staticmethod
    def _derive_dataset_name(dataset: UniversalDataset) -> str:
        """
        Derive a display name for a UniversalDataset for rules that
        need a single dataset-name string (e.g. check_dataset_name).

        A UniversalDataset can hold images merged from several source
        datasets at once (dataset_counts tracks each source's image
        count), so there isn't always a single obvious "name" --
        return the one source name if there's exactly one, or a
        joined name if several were merged together.
        """

        counts = getattr(dataset, "dataset_counts", None) or {}

        names = sorted(counts.keys())

        if len(names) == 1:
            return names[0]

        if len(names) > 1:
            return "+".join(names)

        return ""

    # -------------------------------------------------------------------------
    # Dataset Validation
    # -------------------------------------------------------------------------

    def _validate_dataset(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Execute all dataset validation rules.

        Runs once for the whole dataset (not per-image). The
        filesystem-level checks in dataset_rules.py (does the
        directory exist, are there raw image/annotation files, etc.)
        only run if `self.dataset_path` was provided -- otherwise
        they're skipped gracefully by `_execute_rules` rather than
        forced against data that doesn't exist for an in-memory-only
        UniversalDataset.
        """

        logger.info("Running dataset validation.")

        context = {
            "result": self.result,
            "dataset_name": self._derive_dataset_name(dataset),
        }

        if self.dataset_path is not None:
            context["dataset_path"] = self.dataset_path

        self._execute_rules(
            module=dataset_rules,
            category="dataset",
            context=context,
        )

    # -------------------------------------------------------------------------
    # Image Validation
    # -------------------------------------------------------------------------

    def _validate_images(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Execute image validation rules for every image.
        """

        logger.info("Running image validation.")

        for image in self._iter_images(dataset):

            context = {
                "result": self.result,
                "image_path": str(image.image_path),
                "dataset": image.dataset,
                "seen_paths": self._seen_image_paths,
                "min_width": self.min_image_width,
                "min_height": self.min_image_height,
                "max_width": self.max_image_width,
                "max_height": self.max_image_height,
            }

            self._execute_rules(
                module=image_rules,
                category="image",
                context=context,
            )

            if (

                self.stop_on_error

                and not self.result.passed

            ):

                logger.warning(
                    "Validation stopped after image validation."
                )

                return

    # -------------------------------------------------------------------------
    # Object Validation
    # -------------------------------------------------------------------------

    def _validate_objects(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Execute object validation rules for every image.

        object_rules.py's checks (e.g. check_duplicate_objects,
        check_single_class_image) operate on the *whole list* of
        objects belonging to one image at once, not one object at a
        time -- so this runs once per image, not once per object.
        """

        logger.info("Running object validation.")

        for image in self._iter_images(dataset):

            objects = [
                self._bbox_to_dict(obj)
                for obj in self._iter_objects(image)
            ]

            context = {
                "result": self.result,
                "objects": objects,
                "dataset": image.dataset,
                "image_path": str(image.image_path),
            }

            # Only included when actually configured -- if left as
            # None, the rules needing them (check_invalid_class_id,
            # check_missing_classes) are skipped by _execute_rules
            # rather than run with a meaningless None.
            if self.valid_class_ids is not None:
                context["valid_class_ids"] = self.valid_class_ids

            if self.expected_classes is not None:
                context["expected_classes"] = self.expected_classes

            self._execute_rules(
                module=object_rules,
                category="object",
                context=context,
            )

            if (

                self.stop_on_error

                and not self.result.passed

            ):

                logger.warning(
                    "Validation stopped after object validation."
                )

                return

    # -------------------------------------------------------------------------
    # Bounding Box Validation
    # -------------------------------------------------------------------------

    def _validate_bboxes(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Execute bounding-box validation rules for every image.

        Like object_rules.py, bbox_rules.py's checks (e.g.
        check_duplicate_bboxes, check_overlapping_bboxes) operate on
        the whole list of an image's boxes at once -- runs once per
        image, not once per box.
        """

        logger.info("Running bounding-box validation.")

        for image in self._iter_images(dataset):

            objects = [
                self._bbox_to_dict(obj)
                for obj in self._iter_objects(image)
            ]

            context = {
                "result": self.result,
                "objects": objects,
                "dataset": image.dataset,
                "image_path": str(image.image_path),
                "image_width": getattr(image, "width", None),
                "image_height": getattr(image, "height", None),
                # BoundingBox stores normalized center/width/height,
                # which is exactly bbox_rules.py's "yolo" format.
                "bbox_format": "yolo",
                "iou_threshold": self.iou_threshold,
                "min_size": self.min_bbox_size,
                "max_ratio": self.max_bbox_aspect_ratio,
            }

            self._execute_rules(
                module=bbox_rules,
                category="bounding_box",
                context=context,
            )

            if (

                self.stop_on_error

                and not self.result.passed

            ):

                logger.warning(
                    "Validation stopped after bounding-box validation."
                )

                return

    # -------------------------------------------------------------------------
    # Finalization
    # -------------------------------------------------------------------------

    def _finalize(
        self,
        dataset: UniversalDataset,
    ) -> None:
        """
        Finalize the validation process.

        Responsibilities
        ----------------
        • Compute dataset statistics
        • Compute quality metrics
        • Record execution metadata
        """

        logger.info("Computing dataset statistics.")

        self.result.statistics = compute_statistics(
            dataset
        )

        logger.info("Computing quality metrics.")

        self.result.metrics = compute_metrics(
            dataset,
            self.result,
        )

        self.result.metadata.mark_completed()

        self.result.summary.validation_time = (
            self.result.metadata.execution_time
        )

        logger.info(
            "Validation completed in %.3f seconds.",
            self.result.metadata.execution_time,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def validate(
        self,
        dataset: UniversalDataset,
    ) -> ValidationResult:
        """
        Validate a UniversalDataset.

        Parameters
        ----------
        dataset : UniversalDataset
            Dataset to validate.

        Returns
        -------
        ValidationResult
            Complete validation result containing:

            • Summary
            • Validation Issues
            • Dataset Statistics
            • Quality Metrics
            • Validation Metadata
        """

        if not isinstance(
            dataset,
            UniversalDataset,
        ):
            raise TypeError(
                "Expected UniversalDataset."
            )

        logger.info("Starting dataset validation.")

        self._reset()

        self._initialize(dataset)

        # -------------------------------------------------
        # Dataset Validation
        # -------------------------------------------------

        self._validate_dataset(dataset)

        if (
            self.stop_on_error
            and not self.result.passed
        ):
            self._finalize(dataset)
            return self.result

        # -------------------------------------------------
        # Image Validation
        # -------------------------------------------------

        if self.validate_images:

            self._validate_images(dataset)

            if (
                self.stop_on_error
                and not self.result.passed
            ):
                self._finalize(dataset)
                return self.result

        # -------------------------------------------------
        # Object Validation
        # -------------------------------------------------

        if self.validate_objects:

            self._validate_objects(dataset)

            if (
                self.stop_on_error
                and not self.result.passed
            ):
                self._finalize(dataset)
                return self.result

        # -------------------------------------------------
        # Bounding Box Validation
        # -------------------------------------------------

        if self.validate_bboxes:

            self._validate_bboxes(dataset)

            if (
                self.stop_on_error
                and not self.result.passed
            ):
                self._finalize(dataset)
                return self.result

        # -------------------------------------------------
        # Statistics / Metrics
        # -------------------------------------------------

        self._finalize(dataset)

        logger.info(
            "Dataset validation finished successfully."
        )

        return self.result
    # -------------------------------------------------------------------------
    # Rule Cache
    # -------------------------------------------------------------------------

    def _build_rule_cache(self) -> None:
        """
        Discover validation rules once and cache them.

        This avoids repeatedly scanning rule modules during
        validation and improves performance on large datasets.
        """

        self._rule_cache = {

            "dataset": self._discover_rules(dataset_rules),

            "image": self._discover_rules(image_rules),

            "object": self._discover_rules(object_rules),

            "bbox": self._discover_rules(bbox_rules),

        }

    # -------------------------------------------------------------------------
    # Validation Configuration
    # -------------------------------------------------------------------------

    @property
    def configuration(self) -> dict:
        """
        Return the validator configuration.
        """

        return {

            "stop_on_error": self.stop_on_error,

            "validate_images": self.validate_images,

            "validate_objects": self.validate_objects,

            "validate_bboxes": self.validate_bboxes,

        }

    # -------------------------------------------------------------------------
    # Rule Information
    # -------------------------------------------------------------------------

    def list_rules(self) -> dict:
        """
        Return all discovered validation rules.

        Useful for debugging, documentation,
        and unit testing.
        """

        if not hasattr(self, "_rule_cache"):

            self._build_rule_cache()

        return {

            key: [

                rule.__name__

                for rule in rules

            ]

            for key, rules in self._rule_cache.items()

        }

    # -------------------------------------------------------------------------
    # Validator Information
    # -------------------------------------------------------------------------

    @property
    def validator_name(self) -> str:
        return "BloodCellAI Dataset Validator"

    @property
    def validator_version(self) -> str:
        return "4.0.0"

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:

        return (

            f"{self.__class__.__name__}("

            f"stop_on_error={self.stop_on_error}, "

            f"validate_images={self.validate_images}, "

            f"validate_objects={self.validate_objects}, "

            f"validate_bboxes={self.validate_bboxes}"

            ")"

        )

    def __str__(self) -> str:

        return (

            f"{self.validator_name} "

            f"v{self.validator_version}"

        )
