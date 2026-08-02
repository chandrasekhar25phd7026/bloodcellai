"""
BloodCellAI reproducibility scripts -- 03: Main cross-dataset benchmark
(Table 5, Figures 4-5, Supplementary Table 6).

Builds all 6 datasets, trains one detection or classification model per
dataset at a given seed, and appends results to a shared JSON file so the
script is safe to re-run after an interruption -- already-completed
(dataset, seed) pairs are skipped automatically.

Run 01_setup.py and 02_data_preparation.py (for this seed) first.
Run once per seed: 42, 123, 456, 789, 999.
"""
import os, json, time, gc
from pathlib import Path
from bloodcell.dataset_builder import UniversalDatasetBuilder
from bloodcell.dataset_info import DatasetInfo
from bloodcell.dataset_registry import DATASET_REGISTRY
from bloodcell.universal_builder import UniversalBuilder
from bloodcell.quality_gate import DatasetQualityGate
from bloodcell.dataset_export import export_yolo, export_classification_folders
from ultralytics import YOLO

SEED = 42  # change per session: 42, 123, 456, 789, 999
RESULTS_FILE = "/kaggle/working/repeat_runs_results.json"

EPOCH_BUDGET = {
    "BCCD": 60, "Chula_RBC": 60, "Malaria_BBoxes": 40,
    "NIH_Malaria": 20, "AcuteLeukemia": 30, "Raabin_WBC": 20,
}

CHULA_CLASSES = {
    0: "Normal_RBC", 1: "Macrocyte", 2: "Microcyte", 3: "Spherocyte",
    4: "Target_Cell", 5: "Stomatocyte", 6: "Ovalocyte", 7: "Tear_Drop_Cell",
    8: "Burr_Cell", 9: "Schistocyte", 10: "Uncategorized",
    11: "Hypochromia", 12: "Elliptocyte",
}


def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_result(result):
    results = load_existing_results()
    key = (result["dataset"], result["seed"])
    results = [r for r in results if (r["dataset"], r["seed"]) != key]
    results.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"--> Saved {result['dataset']} seed={result['seed']}")


def build_once(built_datasets, name, path, task, manual_registry=None):
    DATASET_REGISTRY.pop(name, None)
    if manual_registry:
        info = DatasetInfo(id=f"MANUAL-{name}", name=name, task=manual_registry["task"],
                            annotation=manual_registry["annotation"], path=Path(path),
                            registry=manual_registry)
    else:
        db = UniversalDatasetBuilder()
        info = db.auto_register(path, dataset_name=name)
        db.prepare(name)

    ub = UniversalBuilder("/kaggle/working")
    ub.enable_preprocessing()
    dataset, built = ub.build_from_info(info)
    gate = DatasetQualityGate(minimum_quality_score=50.0)
    clean, _ = gate.filter_passing(dataset)
    built_datasets[name] = (clean, task)
    del dataset
    gc.collect()


def run_one(name, clean_dataset, task, seed, epochs):
    result = {"dataset": name, "task": task, "seed": seed}
    t0 = time.time()
    model = None
    try:
        clean_dataset.assign_splits(train=0.8, val=0.1, test=0.1, seed=seed, stratify_by=None)
        export_dir = f"/kaggle/working/repeat_exports/{name}_seed{seed}"

        if task == "detect":
            export_yolo(clean_dataset, export_dir)
            model = YOLO("yolov8n.pt")
            model.train(data=f"{export_dir}/data.yaml", epochs=epochs, imgsz=640,
                        batch=8, workers=2, project="/kaggle/working/repeat_runs",
                        name=f"{name}_seed{seed}", patience=15, device=0,
                        verbose=False, seed=seed)
            metrics = model.val(data=f"{export_dir}/data.yaml", split="test")
            result.update(mAP50=float(metrics.box.map50), **{"mAP50-95": float(metrics.box.map)},
                          precision=float(metrics.box.mp), recall=float(metrics.box.mr))
        else:
            export_classification_folders(clean_dataset, export_dir)
            model = YOLO("yolov8n-cls.pt")
            model.train(data=export_dir, epochs=epochs, imgsz=224, batch=16, workers=2,
                        project="/kaggle/working/repeat_runs", name=f"{name}_seed{seed}",
                        patience=15, device=0, verbose=False, seed=seed)
            metrics = model.val(data=export_dir, split="test")
            result.update(top1_accuracy=float(metrics.top1), top5_accuracy=float(metrics.top5))

        result["status"] = "success"
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    finally:
        del model
        gc.collect()

    result["elapsed_minutes"] = round((time.time() - t0) / 60, 1)
    save_result(result)
    return result


if __name__ == "__main__":
    from importlib import import_module
    prep = import_module("02_data_preparation")
    malaria_resized, nih_capped, leukemia_capped, raabin_capped = prep.prepare_all(SEED)

    built_datasets = {}
    build_once(built_datasets, "BCCD", "/kaggle/input/datasets/coder98/bccd-dataset/BCCD", "detect")
    build_once(built_datasets, "Chula_RBC", "/kaggle/working/Chula-RBC-12-Dataset", "detect",
               manual_registry={"task": "Detection", "annotation": "Chula TXT", "classes": CHULA_CLASSES})
    build_once(built_datasets, "Malaria_BBoxes", malaria_resized, "detect",
               manual_registry={"task": "Detection", "annotation": "Malaria JSON",
                                 "classes": {0: "RBC", 1: "WBC", 3: "Parasite"}})
    build_once(built_datasets, "NIH_Malaria", nih_capped, "classify")
    build_once(built_datasets, "AcuteLeukemia", leukemia_capped, "classify")
    build_once(built_datasets, "Raabin_WBC", raabin_capped, "classify")

    already_done = {(r["dataset"], r["seed"]) for r in load_existing_results() if r.get("status") == "success"}
    for name, (clean_dataset, task) in built_datasets.items():
        if (name, SEED) in already_done:
            print(f"Skipping {name} seed={SEED} -- already done")
            continue
        result = run_one(name, clean_dataset, task, SEED, EPOCH_BUDGET[name])
        print(result)

    print("ALL DONE for seed", SEED)
