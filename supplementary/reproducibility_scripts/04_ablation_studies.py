"""
BloodCellAI reproducibility scripts -- 04: Ablation studies, repeated
across 5 seeds (Table 3, Table 4, Supplementary Table 7).

Ablation A (quality gate): builds a 50%-degraded copy of BCCD and trains
one YOLOv8n model with the quality gate disabled and one with it enabled,
at each seed.

Ablation B (preprocessing): trains one YOLOv8n model each on raw,
resize+normalize, and resize-only exports of clean BCCD, at each seed.

Run 01_setup.py first. This script builds BCCD itself, so 02/03 are not
required first.
"""
import os, json, time, gc
from pathlib import Path
from bloodcell.dataset_builder import UniversalDatasetBuilder
from bloodcell.universal_builder import UniversalBuilder
from bloodcell.quality_gate import DatasetQualityGate
from bloodcell.dataset_export import export_yolo
from preprocessing.preprocessing_config import PreprocessingConfig
from training.degradation import IMAGE_QUALITY_ONLY_SEVERE, create_degraded_copy
from ultralytics import YOLO

RESULTS_FILE = "/kaggle/working/ablation_multiseed_results.json"
SEEDS = [42, 123, 456, 789, 999]


def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_result(result):
    results = load_existing_results()
    key = (result["ablation"], result["condition"], result["seed"])
    results = [r for r in results if (r["ablation"], r["condition"], r["seed"]) != key]
    results.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"--> Saved {result['ablation']} / {result['condition']} / seed={result['seed']}")


def build_bccd():
    db = UniversalDatasetBuilder()
    info = db.auto_register("/kaggle/input/datasets/coder98/bccd-dataset/BCCD",
                             dataset_name="BCCD_ablation_multiseed")
    db.prepare("BCCD_ablation_multiseed")
    ub = UniversalBuilder("/kaggle/working")
    dataset, built = ub.build_from_info(info)
    print("Built:", built)
    return dataset


def run_quality_gate_ablation(dataset, already_done):
    for seed in SEEDS:
        dataset.assign_splits(train=0.8, val=0.1, test=0.1, seed=seed)
        degraded = create_degraded_copy(
            dataset, IMAGE_QUALITY_ONLY_SEVERE, f"/kaggle/working/ablation_degraded_seed{seed}",
            seed=seed, exclude_split="test"
        )

        if ("quality_gate", "off", seed) not in already_done:
            t0 = time.time()
            export_yolo(degraded, f"/kaggle/working/qgate_off_seed{seed}/train_data")
            model = YOLO("yolov8n.pt")
            model.train(data=f"/kaggle/working/qgate_off_seed{seed}/train_data/data.yaml",
                        epochs=60, imgsz=640, batch=8, workers=2,
                        project="/kaggle/working/ablation_runs", name=f"qgate_off_s{seed}",
                        patience=15, device=0, verbose=False, seed=seed)
            metrics = model.val(data=f"/kaggle/working/qgate_off_seed{seed}/train_data/data.yaml", split="test")
            save_result({"ablation": "quality_gate", "condition": "off", "seed": seed,
                         "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
                         "elapsed_minutes": round((time.time() - t0) / 60, 1)})
            del model
            gc.collect()

        if ("quality_gate", "on", seed) not in already_done:
            t0 = time.time()
            gate = DatasetQualityGate(minimum_quality_score=50.0)
            clean, _ = gate.filter_passing(degraded)
            export_yolo(clean, f"/kaggle/working/qgate_on_seed{seed}/train_data")
            model = YOLO("yolov8n.pt")
            model.train(data=f"/kaggle/working/qgate_on_seed{seed}/train_data/data.yaml",
                        epochs=60, imgsz=640, batch=8, workers=2,
                        project="/kaggle/working/ablation_runs", name=f"qgate_on_s{seed}",
                        patience=15, device=0, verbose=False, seed=seed)
            metrics = model.val(data=f"/kaggle/working/qgate_on_seed{seed}/train_data/data.yaml", split="test")
            save_result({"ablation": "quality_gate", "condition": "on", "seed": seed,
                         "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
                         "elapsed_minutes": round((time.time() - t0) / 60, 1)})
            del model
            gc.collect()


def run_preprocessing_ablation(dataset, already_done):
    for seed in SEEDS:
        dataset.assign_splits(train=0.8, val=0.1, test=0.1, seed=seed)

        resize_only_cfg = PreprocessingConfig()
        resize_only_cfg.normalize.enabled = False
        conditions = {"raw": None, "resize_normalize": PreprocessingConfig(), "resize_only": resize_only_cfg}

        for cond_name, config in conditions.items():
            if ("preprocessing", cond_name, seed) in already_done:
                continue
            t0 = time.time()
            export_dir = f"/kaggle/working/preprocess_{cond_name}_seed{seed}/train_data"
            if config is None:
                export_yolo(dataset, export_dir)
            else:
                export_yolo(dataset, export_dir, preprocessing_config=config)
            model = YOLO("yolov8n.pt")
            model.train(data=f"{export_dir}/data.yaml", epochs=60, imgsz=640, batch=8, workers=2,
                        project="/kaggle/working/ablation_runs", name=f"prep_{cond_name}_s{seed}",
                        patience=15, device=0, verbose=False, seed=seed)
            metrics = model.val(data=f"{export_dir}/data.yaml", split="test")
            save_result({"ablation": "preprocessing", "condition": cond_name, "seed": seed,
                         "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
                         "elapsed_minutes": round((time.time() - t0) / 60, 1)})
            del model
            gc.collect()


if __name__ == "__main__":
    dataset = build_bccd()
    already_done = {(r["ablation"], r["condition"], r["seed"]) for r in load_existing_results()}
    run_quality_gate_ablation(dataset, already_done)
    already_done = {(r["ablation"], r["condition"], r["seed"]) for r in load_existing_results()}
    run_preprocessing_ablation(dataset, already_done)
    print("Both ablations complete across all 5 seeds")
