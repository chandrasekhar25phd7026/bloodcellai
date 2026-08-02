
"""
==============================================================
BloodCellAI Universal Dataset Builder
==============================================================

File:
    dataset_builder.py

Description
-----------
Registers datasets (known, via DATASET_REGISTRY, or arbitrary
folders via auto-detection) and orchestrates the pre-build
filesystem-level pipeline for each one:

    register -> scan -> validate

This is the "point BloodCellAI at a folder" entry point: register()
requires a name already present in DATASET_REGISTRY, while
auto_register() accepts *any* folder and uses
annotation_intelligence.detect_dataset_format() to guess its task
and annotation format on the fly, so datasets don't need to be
pre-registered by name to be usable.

The actual annotation-parsing/harmonization into a UniversalDataset
happens downstream, in UniversalBuilder (universal_builder.py) --
this class's job ends at "here is a validated, format-identified
DatasetInfo, ready to be built."

Version:
    2.0.0
"""

import logging

from pathlib import Path

from .dataset_registry import DATASET_REGISTRY
from .dataset_info import DatasetInfo
from .dataset_scanner import scan_dataset
from .dataset_validator import validate_dataset
from .annotation_intelligence import detect_dataset_format

logger = logging.getLogger(__name__)


class UniversalDatasetBuilder:

    def __init__(self):

        self.datasets = []

    # =========================================================================
    # Registration
    # =========================================================================

    def register_dataset(self, dataset_name, dataset_path):
        """
        Register a dataset that's already known in DATASET_REGISTRY
        by name (e.g. "BCCD"). Use auto_register() instead for a
        folder that isn't pre-registered.
        """

        if dataset_name not in DATASET_REGISTRY:

            logger.warning(
                "Skipping %r: not found in DATASET_REGISTRY. "
                "Use auto_register() to register an arbitrary "
                "folder without a prior registry entry.",
                dataset_name,
            )

            return None

        info = DATASET_REGISTRY[dataset_name]

        ds = DatasetInfo(
            id=info["id"],
            name=dataset_name,
            task=info["task"],
            annotation=info["annotation"],
            path=Path(dataset_path),
            registry=info,
        )

        self.datasets.append(ds)

        logger.info("Registered dataset %r from %s", dataset_name, dataset_path)

        return ds

    def auto_register(self, dataset_path, dataset_name=None):
        """
        Point at an arbitrary folder and register it for building,
        WITHOUT requiring a prior DATASET_REGISTRY entry.

        Runs annotation_intelligence.detect_dataset_format() to guess
        the task (Detection/Classification) and annotation format
        (YOLO/COCO/Pascal VOC/CSV/None/Unknown), and builds a
        registry-shaped info dict on the fly from the result.

        Parameters
        ----------
        dataset_path : str or Path
        dataset_name : str, optional
            Defaults to the folder's own name.

        Returns
        -------
        DatasetInfo
        """

        dataset_path = Path(dataset_path)

        dataset_name = dataset_name or dataset_path.name

        detection = detect_dataset_format(dataset_path)

        if detection["annotation"] == "Unknown":

            logger.warning(
                "Could not confidently detect a format for %r (%s). "
                "Evidence considered: %s. Proceeding anyway with "
                "task=Unknown -- downstream scanning/validation will "
                "still run, but no adapter will be able to parse "
                "annotations until this is registered manually or "
                "detection is improved for this layout.",
                dataset_name, dataset_path, detection["evidence"],
            )

        registry_entry = {
            "id": f"AUTO-{dataset_name}",
            "task": detection["task"],
            "annotation": detection["annotation"],
            "image_extension": detection["image_extension"],
            "label_extension": detection["label_extension"],
            "classes": detection["classes"],
            "source": "auto-detected",
            "is_roboflow_export": detection["is_roboflow_export"],
            "detection_evidence": detection["evidence"],
        }

        ds = DatasetInfo(
            id=registry_entry["id"],
            name=dataset_name,
            task=registry_entry["task"],
            annotation=registry_entry["annotation"],
            path=dataset_path,
            registry=registry_entry,
        )

        self.datasets.append(ds)

        logger.info(
            "Auto-registered %r from %s as task=%s, annotation=%s%s",
            dataset_name, dataset_path, registry_entry["task"],
            registry_entry["annotation"],
            " (Roboflow export)" if registry_entry["is_roboflow_export"] else "",
        )

        return ds

    # =========================================================================
    # Lookup
    # =========================================================================

    def get_dataset(self, dataset_name):
        """
        Return the registered DatasetInfo for `dataset_name`, or None.
        """

        for ds in self.datasets:
            if ds.name == dataset_name:
                return ds

        return None

    # =========================================================================
    # Orchestration: scan + validate every registered dataset
    # =========================================================================

    def prepare(self, dataset_name):
        """
        Run the pre-build filesystem pipeline (scan -> validate) for
        one already-registered dataset.

        Returns
        -------
        DatasetInfo
            The same object, now populated with image_files,
            annotation_files, counts, warnings/errors, and status.
        """

        ds = self.get_dataset(dataset_name)

        if ds is None:
            raise ValueError(
                f"{dataset_name!r} is not registered -- call "
                "register_dataset() or auto_register() first."
            )

        logger.info("Scanning %r at %s", ds.name, ds.path)
        scan_dataset(ds)

        logger.info(
            "Validating %r (%d images, %d annotation files found)",
            ds.name, ds.image_count, ds.annotation_count,
        )
        validate_dataset(ds)

        logger.info(
            "%r prepared: status=%s, %d warning(s), %d error(s)",
            ds.name, ds.status, len(ds.warnings), len(ds.errors),
        )

        return ds

    def prepare_all(self):
        """
        Run prepare() for every registered dataset. Continues past
        individual failures (logged, not raised) so one bad dataset
        doesn't block preparing the rest of a multi-dataset study.
        """

        results = {}

        for ds in list(self.datasets):

            try:
                results[ds.name] = self.prepare(ds.name)

            except Exception:

                logger.exception(
                    "Failed to prepare dataset %r", ds.name
                )

                ds.status = "ERROR"
                ds.errors.append("Preparation raised an exception -- see log.")

                results[ds.name] = ds

        return results

    # =========================================================================
    # Reporting
    # =========================================================================

    def summary(self):
        """
        Human-readable summary of every registered dataset. Returns
        the summary text (also logged) so callers can capture it
        rather than only printing to stdout.
        """

        lines = [
            "=" * 80,
            "Registered Enterprise Datasets",
            "=" * 80,
        ]

        for ds in self.datasets:

            lines.append(
                f"{ds.id:10} {ds.name:30} {ds.task:15} "
                f"{ds.annotation:15} status={ds.status}"
            )

        lines.append("")
        lines.append(f"TOTAL : {len(self.datasets)}")

        text = "\n".join(lines)

        print(text)

        return text
