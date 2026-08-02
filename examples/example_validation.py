"""
BloodCellAI example: run the validation engine and inspect a dataset's
Blood Dataset Quality Index (BDQI).

Usage:
    python example_validation.py /path/to/your/dataset
"""
import sys

from bloodcell.dataset_builder import UniversalDatasetBuilder
from bloodcell.universal_builder import UniversalBuilder
from validation.validator import DatasetValidatorV2


def main(dataset_path: str):
    builder = UniversalDatasetBuilder()
    info = builder.auto_register(dataset_path)
    builder.prepare(info.name)

    ub = UniversalBuilder("/tmp/bloodcellai_validation_example")
    dataset, built = ub.build_from_info(info)
    dataset.assign_splits(train=0.8, val=0.1, test=0.1, seed=42)
    print(f"Built {built} images from {dataset_path}")

    # Run the validation engine and print the full report
    validator = DatasetValidatorV2()
    report = validator.validate(dataset)

    print("\n--- Validation Summary ---")
    print(f"Total images:        {report.summary.total_images}")
    print(f"Total objects:       {report.summary.total_objects}")
    print(f"Passed:              {report.passed}")
    print(f"Errors / Warnings:   {report.error_count} / {report.warning_count}")

    print("\n--- BDQI Component Scores ---")
    print(f"Annotation completeness:  {report.metrics.annotation_completeness_score:.2f}")
    print(f"Image integrity:          {report.metrics.image_integrity_score:.2f}")
    print(f"Class consistency:        {report.metrics.class_consistency_score:.2f}")
    print(f"Bounding-box validity:    {report.metrics.bounding_box_validity_score:.2f}")
    print(f"Overall grade:            {report.metrics.overall_grade}")
    print(f"\nBDQI (composite score):   {report.metrics.bdqi_score:.2f} / 100")

    # A dataset's specific list of issues (useful for debugging low scores)
    if report.issues:
        print(f"\nFirst 5 of {len(report.issues)} issues found:")
        for issue in report.issues[:5]:
            print(f"  - {issue}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python example_validation.py <dataset_path>")
        sys.exit(1)
    main(sys.argv[1])

# ---------------------------------------------------------------------
# Note: the exact attribute names above (report.metrics.bdqi_score, etc.)
# were confirmed directly against a real run of this codebase during this
# project's own BDQI validation experiments (see the accompanying paper's
# Technical Validation section and supplementary reproducibility scripts).
# If you have modified the validation engine, re-check these attribute
# names against your own version.
# ---------------------------------------------------------------------
