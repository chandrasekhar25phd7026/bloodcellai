"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    reports.py

Version:
    4.0.0

Purpose:
    Dataset Validation Report Generator

Description:
    Generates professional validation reports from a ValidationResult.

Supported Outputs
-----------------
✓ Text Report
✓ Dictionary Report
✓ JSON Report

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .models import (
    ValidationResult,
    ValidationIssue,
)


# =============================================================================
# Dataset Report Engine
# =============================================================================

class DatasetReportEngine:
    """
    Generates professional validation reports.

    This module performs no validation.

    Inputs
    ------
    ValidationResult

    Outputs
    -------
    Text Report
    Dictionary
    JSON
    """

    def __init__(self):

        self.result = None

        self.lines: List[str] = []

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self):

        self.result = None

        self.lines.clear()

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def _add_line(
        self,
        text: str = "",
    ):

        self.lines.append(text)

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def _separator(
        self,
        char: str = "=",
        length: int = 79,
    ):

        self.lines.append(char * length)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate(
        self,
        result: ValidationResult,
    ) -> str:
        """
        Generate a complete validation report.

        Parameters
        ----------
        result : ValidationResult

        Returns
        -------
        str
            Formatted report.
        """

        self.reset()

        self.result = result

        self._header()

        self._metadata()

        self._summary()

        self._statistics()

        self._metrics()

        self._issues()

        self._footer()

        return "\n".join(self.lines)

    # -------------------------------------------------------------------------
    # Report Sections
    # -------------------------------------------------------------------------

    def _header(self):
        pass

    def _metadata(self):
        pass

    def _summary(self):
        pass

    def _statistics(self):
        pass

    def _metrics(self):
        pass

    def _issues(self):
        pass

    def _footer(self):
        pass
    # -------------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------------

    def _header(self) -> None:
        """
        Report title.
        """

        self._separator()

        self._add_line(
            "BloodCellAI Dataset Validation Report"
        )

        self._separator()

        self._add_line()

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def _metadata(self) -> None:
        """
        Validation metadata.
        """

        metadata = self.result.metadata

        self._add_line("Metadata")
        self._separator("-")

        self._add_line(
            f"Framework          : {metadata.framework_name}"
        )

        self._add_line(
            f"Version            : {metadata.framework_version}"
        )

        self._add_line(
            f"Dataset Name       : {metadata.dataset_name}"
        )

        self._add_line(
            f"Dataset Version    : {metadata.dataset_version}"
        )

        self._add_line(
            f"Dataset Path       : {metadata.dataset_path}"
        )

        self._add_line(
            f"Started At         : {metadata.started_at}"
        )

        self._add_line(
            f"Completed At       : {metadata.completed_at}"
        )

        self._add_line(
            f"Execution Time     : "
            f"{metadata.execution_time:.2f} sec"
        )

        self._add_line()

    # -------------------------------------------------------------------------
    # Validation Summary
    # -------------------------------------------------------------------------

    def _summary(self) -> None:
        """
        Validation summary.
        """

        summary = self.result.summary

        self._add_line("Validation Summary")

        self._separator("-")

        self._add_line(
            f"Validation Passed  : {summary.passed}"
        )

        self._add_line(
            f"Total Images       : {summary.total_images}"
        )

        self._add_line(
            f"Total Objects      : {summary.total_objects}"
        )

        self._add_line(
            f"Total Issues       : {summary.total_issues}"
        )

        self._add_line(
            f"Information        : {summary.info}"
        )

        self._add_line(
            f"Warnings           : {summary.warnings}"
        )

        self._add_line(
            f"Errors             : {summary.errors}"
        )

        self._add_line(
            f"Critical           : {summary.critical}"
        )

        self._add_line()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def _statistics(self) -> None:
        """
        Dataset statistics.
        """

        stats = self.result.statistics

        self._add_line("Dataset Statistics")

        self._separator("-")

        self._add_line(
            f"Total Images               : {stats.total_images}"
        )

        self._add_line(
            f"Total Objects              : {stats.total_objects}"
        )

        self._add_line(
            f"Bounding Boxes             : "
            f"{stats.total_bounding_boxes}"
        )

        self._add_line()

        self._add_line(
            f"Minimum Width              : "
            f"{stats.minimum_width}"
        )

        self._add_line(
            f"Maximum Width              : "
            f"{stats.maximum_width}"
        )

        self._add_line(
            f"Average Width              : "
            f"{stats.average_width:.2f}"
        )

        self._add_line()

        self._add_line(
            f"Minimum Height             : "
            f"{stats.minimum_height}"
        )

        self._add_line(
            f"Maximum Height             : "
            f"{stats.maximum_height}"
        )

        self._add_line(
            f"Average Height             : "
            f"{stats.average_height:.2f}"
        )

        self._add_line()

        self._add_line(
            f"Average Objects/Image      : "
            f"{stats.average_objects_per_image:.2f}"
        )

        self._add_line(
            f"Minimum Objects/Image      : "
            f"{stats.minimum_objects_per_image}"
        )

        self._add_line(
            f"Maximum Objects/Image      : "
            f"{stats.maximum_objects_per_image}"
        )

        self._add_line(
            f"Empty Images               : "
            f"{stats.empty_images}"
        )

        self._add_line()

        self._add_line(
            f"Minimum Bounding Box Width : "
            f"{stats.minimum_bbox_width:.2f}"
        )

        self._add_line(
            f"Maximum Bounding Box Width : "
            f"{stats.maximum_bbox_width:.2f}"
        )

        self._add_line(
            f"Average Bounding Box Width : "
            f"{stats.average_bbox_width:.2f}"
        )

        self._add_line()

        self._add_line(
            f"Minimum Bounding Box Height: "
            f"{stats.minimum_bbox_height:.2f}"
        )

        self._add_line(
            f"Maximum Bounding Box Height: "
            f"{stats.maximum_bbox_height:.2f}"
        )

        self._add_line(
            f"Average Bounding Box Height: "
            f"{stats.average_bbox_height:.2f}"
        )

        self._add_line()

        self._add_line(
            f"Minimum Bounding Box Area  : "
            f"{stats.minimum_bbox_area:.2f}"
        )

        self._add_line(
            f"Maximum Bounding Box Area  : "
            f"{stats.maximum_bbox_area:.2f}"
        )

        self._add_line(
            f"Average Bounding Box Area  : "
            f"{stats.average_bbox_area:.2f}"
        )

        self._add_line()

        self._add_line(
            f"Dataset Distribution       : "
            f"{stats.dataset_counts}"
        )

        self._add_line(
            f"Class Distribution         : "
            f"{stats.class_counts}"
        )

        self._add_line(
            f"Split Distribution         : "
            f"{stats.split_counts}"
        )

        self._add_line(
            f"Image Formats              : "
            f"{stats.image_formats}"
        )

        self._add_line()
    # -------------------------------------------------------------------------
    # Quality Metrics
    # -------------------------------------------------------------------------

    def _metrics(self) -> None:
        """
        Dataset quality metrics.
        """

        metrics = self.result.metrics

        self._add_line("Quality Metrics")

        self._separator("-")

        self._add_line(
            f"Annotation Completeness Score : "
            f"{metrics.annotation_completeness_score:.2f}"
        )

        self._add_line(
            f"Image Integrity Score         : "
            f"{metrics.image_integrity_score:.2f}"
        )

        self._add_line(
            f"Class Consistency Score       : "
            f"{metrics.class_consistency_score:.2f}"
        )

        self._add_line(
            f"Bounding Box Validity Score   : "
            f"{metrics.bounding_box_validity_score:.2f}"
        )

        self._add_line()

        self._add_line(
            f"Blood Dataset Quality Index   : "
            f"{metrics.blood_dataset_quality_index:.2f}"
        )

        self._add_line(
            f"Overall Grade                 : "
            f"{metrics.overall_grade}"
        )

        self._add_line()

    # -------------------------------------------------------------------------
    # Validation Issues
    # -------------------------------------------------------------------------

    def _issues(self) -> None:
        """
        List all validation issues.
        """

        issues = self.result.issues

        self._add_line("Validation Issues")

        self._separator("-")

        if not issues:

            self._add_line(
                "No validation issues detected."
            )

            self._add_line()

            return

        for index, issue in enumerate(
            issues,
            start=1,
        ):

            self._add_line(
                f"Issue #{index}"
            )

            self._add_line(
                f"Severity       : {issue.severity}"
            )

            self._add_line(
                f"Category       : {issue.category}"
            )

            self._add_line(
                f"Rule           : {issue.rule_name}"
            )

            self._add_line(
                f"Dataset        : {issue.dataset}"
            )

            self._add_line(
                f"Image          : {issue.image_path}"
            )

            self._add_line(
                f"Annotation     : {issue.annotation_path}"
            )

            self._add_line(
                f"Object Index   : {issue.object_index}"
            )

            self._add_line(
                f"Class ID       : {issue.class_id}"
            )

            self._add_line(
                f"Message        : {issue.message}"
            )

            self._add_line(
                f"Recommendation : "
                f"{issue.recommendation}"
            )

            if issue.details:

                self._add_line(
                    f"Details        : {issue.details}"
                )

            self._separator("-")

        self._add_line()

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------

    def _footer(self) -> None:
        """
        Report footer.
        """

        self._separator()

        self._add_line(
            "End of Validation Report"
        )

        self._separator()
# =============================================================================
# Dictionary Export
# =============================================================================

    def to_dict(self) -> dict:
        """
        Convert the validation result into a dictionary.
        """

        if self.result is None:
            raise RuntimeError(
                "No report has been generated."
            )

        return self.result.to_dict()

    # -------------------------------------------------------------------------
    # JSON Export
    # -------------------------------------------------------------------------

    def to_json(
        self,
        indent: int = 4,
    ) -> str:
        """
        Convert the validation result into JSON.
        """

        return json.dumps(
            self.to_dict(),
            indent=indent,
        )

    # -------------------------------------------------------------------------
    # Save Text Report
    # -------------------------------------------------------------------------

    def save_text_report(
        self,
        filename: str | Path,
    ) -> Path:
        """
        Save the generated text report.
        """

        path = Path(filename)

        path.write_text(

            "\n".join(self.lines),

            encoding="utf-8",

        )

        return path

    # -------------------------------------------------------------------------
    # Save JSON Report
    # -------------------------------------------------------------------------

    def save_json_report(
        self,
        filename: str | Path,
        indent: int = 4,
    ) -> Path:
        """
        Save the validation report as JSON.
        """

        path = Path(filename)

        path.write_text(

            self.to_json(indent),

            encoding="utf-8",

        )

        return path


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_report(
    result: ValidationResult,
) -> str:
    """
    Generate a text validation report.
    """

    engine = DatasetReportEngine()

    return engine.generate(result)


def save_text_report(
    result: ValidationResult,
    filename: str | Path,
) -> Path:
    """
    Generate and save a text report.
    """

    engine = DatasetReportEngine()

    engine.generate(result)

    return engine.save_text_report(
        filename,
    )


def save_json_report(
    result: ValidationResult,
    filename: str | Path,
    indent: int = 4,
) -> Path:
    """
    Generate and save a JSON report.
    """

    engine = DatasetReportEngine()

    engine.generate(result)

    return engine.save_json_report(
        filename,
        indent,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [

    "DatasetReportEngine",

    "generate_report",

    "save_text_report",

    "save_json_report",

]