"""
===============================================================================
BloodCellAI Dataset Validation Engine (BDVE)
===============================================================================

Package:
    bloodcell.validation

Version:
    1.1.0

Status:
    Stable

Purpose:
    Validation package for UniversalDataset objects. Provides:
      - DatasetValidator : runs all rule modules end-to-end
      - DatasetStatistics : dataset-level statistics
      - ValidationMetricsCalculator : quality metrics incl. BDQI
      - ValidationReport : Text/Markdown/JSON report generation

Author:
    Sekhar Muthangi

Project:
    BloodCellAI
===============================================================================
"""

from .models import (
    ValidationSeverity,
    ValidationCategory,
    ValidationIssue,
    ValidationSummary,
    ValidationStatistics,
    ValidationMetrics,
    ValidationResult,
)

from .validator import DatasetValidator
from .statistics import DatasetStatistics
from .metrics import ValidationMetricsCalculator
from .reports import ValidationReport

__all__ = [
    "ValidationSeverity",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationSummary",
    "ValidationStatistics",
    "ValidationMetrics",
    "ValidationResult",
    "DatasetValidator",
    "DatasetStatistics",
    "ValidationMetricsCalculator",
    "ValidationReport",
]
