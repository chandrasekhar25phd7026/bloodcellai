"""
BloodCellAI example: build, quality-gate, and export a dataset end to end.

This mirrors the Quickstart example in the main README. It works on any
dataset folder whose format BloodCellAI can auto-detect (Pascal VOC XML,
YOLO plain text, COCO JSON, or folder-per-class classification layouts).
For formats requiring manual registration (point annotations, whole-dataset
JSON), see the comment at the bottom of this file.

Usage:
    python example_build_dataset.py /path/to/your/dataset /path/to/output
"""
import sys
from pathlib import Path

from bloodcell.dataset_builder import UniversalDatasetBuilder
from bloodcell.universal_builder import UniversalBuilder
from bloodcell.quality_gate import DatasetQualityGate
from bloodcell.dataset_export import export_yolo, export_classification_folders


def main(dataset_path: str, output_path: str):
    # Step 1: detect the dataset's format automatically
    builder = UniversalDatasetBuilder()
    info = builder.auto_register(dataset_path)
    builder.prepare(info.name)
    print(f"Detected: task={info.task}, annotation={info.annotation}")

    # Step 2: build the harmonized in-memory dataset, with quality scoring enabled
    ub = UniversalBuilder(output_path)
    ub.enable_preprocessing()
    dataset, built = ub.build_from_info(info)
    print(f"Built {built} images")

    # Step 3: filter out low-quality images before export
    gate = DatasetQualityGate(minimum_quality_score=50.0)
    clean_dataset, report = gate.filter_passing(dataset)
    print(f"Quality gate: {len(clean_dataset)}/{built} images passed "
          f"(pass rate {report.pass_rate:.1%})")

    # Step 4: split and export in a standard, training-ready layout
    clean_dataset.assign_splits(train=0.8, val=0.1, test=0.1, seed=42)

    export_dir = f"{output_path}/export"
    if info.task.lower() == "detection":
        export_yolo(clean_dataset, export_dir)
        print(f"Exported YOLO-format dataset to {export_dir}")
    else:
        export_classification_folders(clean_dataset, export_dir)
        print(f"Exported folder-per-class dataset to {export_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python example_build_dataset.py <dataset_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

# ---------------------------------------------------------------------
# If your dataset's format is NOT auto-detected (e.g. point annotations,
# or a whole-dataset JSON schema), register it manually instead of calling
# auto_register(). For example:
#
#   from bloodcell.dataset_info import DatasetInfo
#   from pathlib import Path
#
#   info = DatasetInfo(
#       id="MANUAL-my_dataset", name="my_dataset", task="Detection",
#       annotation="Point", path=Path(dataset_path),
#       registry={"task": "Detection", "annotation": "Point",
#                 "classes": {0: "ClassA", 1: "ClassB"}},
#   )
#
# Everything from Step 2 onward is identical either way.
# ---------------------------------------------------------------------
