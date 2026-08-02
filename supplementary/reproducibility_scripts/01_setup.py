"""
BloodCellAI reproducibility scripts -- 01: Environment setup.
Run this first in any Kaggle session before any other script in this bundle.
Requires the bloodcellai Kaggle dataset attached as input, plus the specific
source datasets referenced in each downstream script (BCCD, Chula-RBC-12,
Malaria Bounding Boxes, NIH Malaria, Acute Lymphoblastic Leukemia, and the
Raabin_WBC-labeled dataset -- see manuscript Table 1 / References).
"""
import shutil, os, sys, json, logging, time, gc, random
logging.basicConfig(level=logging.WARNING)

src = "/kaggle/input/datasets/sekharmuthangi/bloodcellai"
work = "/kaggle/working"
for pkg in ["bloodcell", "preprocessing", "transforms", "training"]:
    nested = f"{src}/{pkg}/{pkg}"
    flat = f"{src}/{pkg}"
    source_path = nested if os.path.isdir(nested) else flat
    shutil.copytree(source_path, f"{work}/{pkg}", dirs_exist_ok=True)
sys.path.insert(0, work)
os.chdir(work)

os.system("pip install ultralytics -q")

from pathlib import Path
from bloodcell.dataset_builder import UniversalDatasetBuilder
from bloodcell.dataset_info import DatasetInfo
from bloodcell.dataset_registry import DATASET_REGISTRY
from bloodcell.universal_builder import UniversalBuilder
from bloodcell.quality_gate import DatasetQualityGate
from bloodcell.dataset_export import export_yolo, export_classification_folders
from ultralytics import YOLO

with open("/kaggle/working/bloodcell/dataset_export.py") as f:
    assert "original_to_contiguous" in f.read(), "Fix missing -- re-upload bloodcell.zip"
with open("/kaggle/working/bloodcell/universal_builder.py") as f:
    assert "_merged_whole_dataset" in f.read(), "Fix missing -- re-upload bloodcell.zip"
print("Setup complete, fixes confirmed present")
