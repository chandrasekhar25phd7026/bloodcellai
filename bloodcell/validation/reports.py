"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

File:
    reports.py

Version:
    1.1.0

Status:
    Stable

Purpose:
    Generates validation reports in Text, Markdown and JSON formats.

Author:
    Sekhar Muthangi

Project:
    BloodCellAI
===============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path


class ValidationReport:
    """
    Generates reports from a ValidationResult.
    """

    def __init__(self, validation_result):

        self.result = validation_result

    # =========================================================================
    # TEXT REPORT
    # =========================================================================

    def to_text(self) -> str:

        summary = self.result.summary
        stats = self.result.statistics
        metrics = self.result.metrics

        lines = []

        lines.append("=" * 70)
        lines.append("BloodCellAI Validation Report")
        lines.append("=" * 70)

        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------

        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 70)

        lines.append(f"Total Issues      : {summary.total_issues}")
        lines.append(f"Critical          : {summary.critical}")
        lines.append(f"Errors            : {summary.errors}")
        lines.append(f"Warnings          : {summary.warnings}")
        lines.append(f"Information       : {summary.info}")
        lines.append(f"Passed            : {summary.passed}")
        lines.append(f"Validation Time   : {summary.validation_time:.3f} sec")

        # ---------------------------------------------------------------------
        # Dataset Statistics
        # ---------------------------------------------------------------------

        lines.append("")
        lines.append("DATASET STATISTICS")
        lines.append("-" * 70)

        lines.append(f"Images            : {stats.total_images}")
        lines.append(f"Objects           : {stats.total_objects}")
        lines.append(f"Classes           : {stats.number_of_classes}")
        lines.append(f"Objects / Image   : {stats.objects_per_image:.2f}")
        lines.append(f"Average Width     : {stats.average_width:.2f}")
        lines.append(f"Average Height    : {stats.average_height:.2f}")

        if stats.dataset_counts:
            lines.append("")
            lines.append("Dataset Distribution")

            for name, count in sorted(stats.dataset_counts.items()):
                lines.append(f"  {name:<20} {count}")

        if stats.class_counts:
            lines.append("")
            lines.append("Class Distribution")

            for name, count in sorted(stats.class_counts.items()):
                lines.append(f"  {name:<20} {count}")

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        lines.append("")
        lines.append("QUALITY METRICS")
        lines.append("-" * 70)

        lines.append(
            f"Annotation Score  : {metrics.annotation_completeness_score:.2f}"
        )

        lines.append(
            f"Image Score       : {metrics.image_integrity_score:.2f}"
        )

        lines.append(
            f"Class Score       : {metrics.class_consistency_score:.2f}"
        )

        lines.append(
            f"BBox Score        : {metrics.bounding_box_validity_score:.2f}"
        )

        lines.append(
            f"BDQI              : {metrics.bdqi:.2f}"
        )

        # ---------------------------------------------------------------------
        # Issues
        # ---------------------------------------------------------------------

        lines.append("")
        lines.append("VALIDATION ISSUES")
        lines.append("-" * 70)

        if not self.result.issues:

            lines.append("No validation issues found.")

        else:

            for issue in self.result.issues:

                lines.append(
                    f"[{issue.severity.name}] "
                    f"{issue.category.name} | "
                    f"{issue.message}"
                )

                if issue.image_path:
                    lines.append(f"   Image : {issue.image_path}")

                if issue.recommendation:
                    lines.append(
                        f"   Recommendation : {issue.recommendation}"
                    )

                lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # MARKDOWN REPORT
    # =========================================================================

    def to_markdown(self):

        summary = self.result.summary
        stats = self.result.statistics
        metrics = self.result.metrics

        md = []

        md.append("# BloodCellAI Validation Report")

        md.append("")

        md.append("## Summary")

        md.append(f"- Total Issues: {summary.total_issues}")
        md.append(f"- Critical: {summary.critical}")
        md.append(f"- Errors: {summary.errors}")
        md.append(f"- Warnings: {summary.warnings}")
        md.append(f"- Information: {summary.info}")
        md.append(f"- Passed: {summary.passed}")

        md.append("")

        md.append("## Dataset Statistics")

        md.append(f"- Images: {stats.total_images}")
        md.append(f"- Objects: {stats.total_objects}")
        md.append(f"- Classes: {stats.number_of_classes}")
        md.append(f"- Objects/Image: {stats.objects_per_image:.2f}")
        md.append(f"- Average Width: {stats.average_width:.2f}")
        md.append(f"- Average Height: {stats.average_height:.2f}")

        md.append("")

        md.append("## Quality Metrics")

        md.append(
            f"- Annotation Score: {metrics.annotation_completeness_score:.2f}"
        )

        md.append(
            f"- Image Integrity Score: {metrics.image_integrity_score:.2f}"
        )

        md.append(
            f"- Class Consistency Score: {metrics.class_consistency_score:.2f}"
        )

        md.append(
            f"- Bounding Box Score: {metrics.bounding_box_validity_score:.2f}"
        )

        md.append(
            f"- BDQI: {metrics.bdqi:.2f}"
        )

        md.append("")

        md.append("## Validation Issues")

        if not self.result.issues:

            md.append("No validation issues found.")

        else:

            for issue in self.result.issues:

                md.append(
                    f"- **{issue.severity.name}** "
                    f"({issue.category.name}) - "
                    f"{issue.message}"
                )

        return "\n".join(md)

    # =========================================================================
    # DICTIONARY
    # =========================================================================

    def to_dict(self):

        return {

            "summary": self.result.summary.to_dict(),

            "statistics": self.result.statistics.to_dict(),

            "metrics": self.result.metrics.to_dict(),

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

                for issue in self.result.issues

            ],

        }

    # =========================================================================
    # JSON
    # =========================================================================

    def to_json(self, indent=4):

        return json.dumps(
            self.to_dict(),
            indent=indent,
        )

    # =========================================================================
    # SAVE FUNCTIONS
    # =========================================================================

    def save_text(self, filename):

        Path(filename).write_text(
            self.to_text(),
            encoding="utf-8",
        )

    def save_markdown(self, filename):

        Path(filename).write_text(
            self.to_markdown(),
            encoding="utf-8",
        )

    def save_json(self, filename):

        Path(filename).write_text(
            self.to_json(),
            encoding="utf-8",
        )

    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================

    def __repr__(self):

        return "ValidationReport()"