"""
===============================================================================
BloodCellAI Validation Manager
===============================================================================

Coordinates dataset validation, statistics, metrics,
and report generation.

The manager is the public entry point for validating datasets.

Note on fixes applied:
    - Originally imported DatasetValidatorV2 from a sibling module
      `.validator_v2`, which doesn't exist -- DatasetValidatorV2 lives
      in `.validator` (this file used to be physically concatenated
      onto the end of validator.py; they're separate files now).
    - Originally imported UniversalDataset via
      `from ..builder.universal_dataset import UniversalDataset`,
      assuming a nested builder/ subpackage that doesn't match the
      actual bloodcell project layout (see the same note in
      validator.py).
    - Originally called `ValidationReportGenerator().generate_all_reports(...)`,
      a class/method that don't exist anywhere in reports.py. The
      actual reports.py API is `DatasetReportEngine` plus the
      module-level `save_text_report()` / `save_json_report()`
      helpers -- used below instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .validator import DatasetValidatorV2
from .reports import save_text_report, save_json_report
from .models import ValidationResult
from bloodcell.universal_dataset import UniversalDataset

logger = logging.getLogger(__name__)


class ValidationManager:
    """
    High-level interface for dataset validation.

    Responsibilities
    ----------------
    ✓ Run validation
    ✓ Generate reports
    ✓ Export results
    """

    def __init__(
        self,
        validator: Optional[DatasetValidatorV2] = None,
    ):

        self.validator = validator or DatasetValidatorV2()

    # ------------------------------------------------------------------

    def validate(
        self,
        dataset: UniversalDataset,
    ) -> ValidationResult:
        """
        Validate a dataset.
        """

        logger.info("Validation Manager started.")

        result = self.validator.validate(dataset)

        logger.info("Validation Manager completed.")

        return result

    # ------------------------------------------------------------------

    def validate_and_report(
        self,
        dataset: UniversalDataset,
        output_directory: str | Path,
    ) -> ValidationResult:
        """
        Validate dataset and generate reports (text + JSON) into
        `output_directory`.
        """

        result = self.validate(dataset)

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        save_text_report(
            result,
            output_directory / "validation_report.txt",
        )

        save_json_report(
            result,
            output_directory / "validation_report.json",
        )

        return result