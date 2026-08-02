
import os
from pathlib import Path

# BLOODCELL_ROOT can be set as an environment variable to point at wherever
# your dataset/output tree actually lives. If unset, defaults to a
# "BloodCellResearch" folder next to the current working directory, which
# keeps the project portable across machines/OSes (the previous version
# hardcoded a single developer's Windows path here).
PROJECT_ROOT = Path(os.environ.get(
    "BLOODCELL_ROOT",
    Path.cwd() / "BloodCellResearch"
))

DATASET_DIR = PROJECT_ROOT / "datasets"

OUTPUT_DIR = PROJECT_ROOT / "UniversalDetectionDataset"

REPORT_DIR = PROJECT_ROOT / "reports"

LOG_DIR = PROJECT_ROOT / "logs"

CONFIG_DIR = PROJECT_ROOT / "configs"
