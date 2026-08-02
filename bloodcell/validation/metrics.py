"""
BloodCellAI Framework

File:
    metrics.py

Description
-----------
Validation quality metrics and BloodCell Dataset Quality Index (BDQI).

Version:
    1.0.0
"""

from __future__ import annotations

from .utils import safe_divide


class ValidationMetricsCalculator:
    """
    Computes validation quality metrics.
    """

    def __init__(self):

        self.annotation_completeness_score = 0.0

        self.image_integrity_score = 0.0

        self.class_consistency_score = 0.0

        self.bounding_box_validity_score = 0.0

        self.bdqi = 0.0

    ###########################################################################
    # Public API
    ###########################################################################

    def compute(self, validation_result):

        """
        Compute all quality metrics.
        """

        issues = validation_result.issues

        total = len(issues)

        errors = validation_result.summary.errors

        warnings = validation_result.summary.warnings

        critical = validation_result.summary.critical

        # ---------------------------------------------------------------------
        # Annotation Completeness
        # ---------------------------------------------------------------------

        self.annotation_completeness_score = max(
            0.0,
            100.0 - (errors * 5.0)
        )

        # ---------------------------------------------------------------------
        # Image Integrity
        # ---------------------------------------------------------------------

        self.image_integrity_score = max(
            0.0,
            100.0 - ((errors + warnings) * 2.0)
        )

        # ---------------------------------------------------------------------
        # Class Consistency
        # ---------------------------------------------------------------------

        self.class_consistency_score = max(
            0.0,
            100.0 - (warnings * 3.0)
        )

        # ---------------------------------------------------------------------
        # Bounding Box Quality
        # ---------------------------------------------------------------------

        self.bounding_box_validity_score = max(
            0.0,
            100.0 - ((errors + critical) * 4.0)
        )

        # ---------------------------------------------------------------------
        # BloodCell Dataset Quality Index
        # ---------------------------------------------------------------------

        self.bdqi = self.compute_bdqi()

        return self

    ###########################################################################
    # BDQI
    ###########################################################################

    def compute_bdqi(self):

        """
        BloodCell Dataset Quality Index.

        Version 1.0

        Equal weighting of all validation metrics.
        """

        return (

            self.annotation_completeness_score +

            self.image_integrity_score +

            self.class_consistency_score +

            self.bounding_box_validity_score

        ) / 4.0

    ###########################################################################
    # Export
    ###########################################################################

    def to_dict(self):

        return {

            "annotation_completeness_score":
                self.annotation_completeness_score,

            "image_integrity_score":
                self.image_integrity_score,

            "class_consistency_score":
                self.class_consistency_score,

            "bounding_box_validity_score":
                self.bounding_box_validity_score,

            "bloodcell_dataset_quality_index":
                self.bdqi,

        }

    ###########################################################################
    # String Representation
    ###########################################################################

    def __repr__(self):

        return (

            "ValidationMetricsCalculator("

            f"BDQI={self.bdqi:.2f})"

        )