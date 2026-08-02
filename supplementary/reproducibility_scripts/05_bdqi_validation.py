"""
BloodCellAI reproducibility scripts -- 05: BDQI validation experiment
(Table 2, original BCCD validation).

Builds five controlled-degradation copies of a dataset (0%, 10%, 25%, 50%,
75% of training+validation images affected by annotation corruption, image
degradation, and label corruption), computes BDQI for each, trains an
identical YOLOv8n model, and evaluates all five on the same held-out test
split (kept byte-identical across all five copies).

Run 01_setup.py first. Designed to run on BCCD (the original validation
target) or any other detection dataset by changing DATASET_PATH / REGISTRY.
"""
import os, json, time, gc, hashlib
from pathlib import Path
from bloodcell.dataset_builder import UniversalDatasetBuilder
from bloodcell.dataset_info import DatasetInfo
from bloodcell.dataset_registry import DATASET_REGISTRY
from bloodcell.universal_builder import UniversalBuilder
from bloodcell.dataset_export import export_yolo
from training.degradation import DEGRADATION_LEVELS, create_degraded_copy
from validation.validator import DatasetValidatorV2  # BDQI computation
from ultralytics import YOLO

RESULTS_FILE = "/kaggle/working/bdqi_validation_results.json"

# Change these two lines to validate on a different dataset than BCCD:
DATASET_NAME = "BCCD"
DATASET_PATH = "/kaggle/input/datasets/coder98/bccd-dataset/BCCD"
MANUAL_REGISTRY = None  # set to a dict like the Chula_RBC/Malaria_BBoxes registries
                         # in 03_main_benchmark.py if the target dataset needs one


def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_result(result):
    results = load_existing_results()
    key = (result["dataset"], result["level"])
    results = [r for r in results if (r["dataset"], r["level"]) != key]
    results.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"--> Saved {result['dataset']} / {result['level']}")


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def build_dataset():
    DATASET_REGISTRY.pop(DATASET_NAME, None)
    if MANUAL_REGISTRY:
        info = DatasetInfo(id=f"MANUAL-{DATASET_NAME}", name=DATASET_NAME,
                            task=MANUAL_REGISTRY["task"], annotation=MANUAL_REGISTRY["annotation"],
                            path=Path(DATASET_PATH), registry=MANUAL_REGISTRY)
    else:
        db = UniversalDatasetBuilder()
        info = db.auto_register(DATASET_PATH, dataset_name=DATASET_NAME)
        db.prepare(DATASET_NAME)
    ub = UniversalBuilder("/kaggle/working")
    dataset, built = ub.build_from_info(info)
    dataset.assign_splits(train=0.8, val=0.1, test=0.1, seed=42)
    print(f"{DATASET_NAME}: built {built} images")
    return dataset


def run_bdqi_validation(dataset):
    test_paths_before = sorted(img.image_path for img in dataset.test_set())

    already_done = {r["level"] for r in load_existing_results() if r["dataset"] == DATASET_NAME}

    for level in DEGRADATION_LEVELS:  # clean, mild, moderate, severe, extreme
        if level.name in already_done:
            print(f"Skipping {level.name} -- already done")
            continue

        degraded = create_degraded_copy(
            dataset, level, f"/kaggle/working/bdqi_{DATASET_NAME}_{level.name}",
            seed=42, exclude_split="test"
        )

        # Verify test split is genuinely untouched before trusting the result
        test_paths_after = sorted(img.image_path for img in degraded.test_set())
        assert test_paths_before == test_paths_after, "Test split was not held identical!"

        validator = DatasetValidatorV2()
        report = validator.validate(degraded)
        bdqi = report.overall_score

        export_dir = f"/kaggle/working/bdqi_export_{level.name}/train_data"
        export_yolo(degraded, export_dir)

        t0 = time.time()
        model = YOLO("yolov8n.pt")
        model.train(data=f"{export_dir}/data.yaml", epochs=60, imgsz=640, batch=8,
                    workers=2, project="/kaggle/working/bdqi_runs", name=f"{DATASET_NAME}_{level.name}",
                    patience=15, device=0, verbose=False, seed=42)
        metrics = model.val(data=f"{export_dir}/data.yaml", split="test")

        save_result({
            "dataset": DATASET_NAME, "level": level.name,
            "fraction_affected": level.fraction_affected, "bdqi": bdqi,
            "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
            "elapsed_minutes": round((time.time() - t0) / 60, 1),
        })
        del model
        gc.collect()


if __name__ == "__main__":
    dataset = build_dataset()
    run_bdqi_validation(dataset)

    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    results = [r for r in load_existing_results() if r["dataset"] == DATASET_NAME]
    bdqi_vals = [r["bdqi"] for r in results]
    map_vals = [r["mAP50"] for r in results]
    print("Pearson r:", pearsonr(bdqi_vals, map_vals))
    print("Spearman rho:", spearmanr(bdqi_vals, map_vals))
