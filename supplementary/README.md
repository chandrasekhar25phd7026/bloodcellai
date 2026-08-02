# BloodCellAI — Supplementary Materials

This bundle accompanies the manuscript "BloodCellAI: An Automated
Format-Detection, Quality-Assessment, and Harmonization Framework for
Multi-Source Blood Cell Imaging Datasets."

## Contents

### `data/`
- `benchmark_30runs.json` — complete raw results for the main cross-dataset
  benchmark (Table 5, Figures 4-5, Supplementary Table 6): 6 datasets × 5
  seeds (42, 123, 456, 789, 999) = 30 runs. Every field (mAP50, mAP50-95,
  precision, recall, top-1, top-5, elapsed_minutes) is populated with no
  gaps.
- `ablation_25runs.json` — complete raw results for both ablation studies,
  repeated across the same 5 seeds (Table 3, Table 4, Supplementary Table
  7): 10 quality-gate runs + 15 preprocessing runs = 25 runs.
- `bdqi_validation_bccd.json` / `bdqi_validation_chularbc.json` — complete
  raw results for the BDQI validation experiment (Table 2, Table 2b,
  Supplementary Table 8): 5 controlled-degradation levels each, on BCCD
  and Chula_RBC respectively.

### `reproducibility_scripts/`
Numbered, standalone Python scripts reproducing every experiment in the
manuscript, intended to be run as Kaggle notebook cells (each script's
docstring notes any Kaggle-specific path assumptions):

1. `01_setup.py` — environment setup, run first in any session.
2. `02_data_preparation.py` — per-seed capped/resized data preparation for
   the three larger classification datasets and Malaria_BBoxes.
3. `03_main_benchmark.py` — the main cross-dataset benchmark, one call per
   seed. Produces `benchmark_30runs.json`-equivalent output.
4. `04_ablation_studies.py` — both ablation studies across all 5 seeds.
   Produces `ablation_25runs.json`-equivalent output.
5. `05_bdqi_validation.py` — the BDQI validation experiment (controlled
   degradation vs. detection performance). Configured for BCCD by default;
   change `DATASET_NAME` / `DATASET_PATH` / `MANUAL_REGISTRY` to validate
   on a different dataset.

Each script is idempotent and safe to re-run after an interruption:
already-completed (dataset, seed) or (ablation, condition, seed)
combinations are detected from the existing results file and skipped
automatically, matching the crash-resilient pattern used throughout this
study's actual Kaggle sessions.

## Reproducing a specific number in the paper

Every number in Tables 2-7 traces back to exactly one row in one of the
two JSON files above, or (for Table 2, the original single-dataset BDQI
validation) to a run of `05_bdqi_validation.py`. To reproduce any single
cell: identify the (dataset, seed) or (ablation, condition, seed) key,
find the matching script, and run it standalone -- no other part of the
pipeline needs to be re-run first beyond `01_setup.py` and, for the main
benchmark, `02_data_preparation.py` for that seed.


## Correction note

An earlier draft of this study referred to one dataset as "Raabin_WBC"
based on an initial, incorrect assumption about its origin. Final
reference verification confirmed this dataset (Kaggle,
paultimothymooney/blood-cells, by Paul Mooney) has no connection to the
actual Raabin-WBC dataset (Kouzehkanan et al., 2022). It is referred to
throughout this bundle and the manuscript as `Mooney_WBC`. This file
naming and the `dataset` field in `benchmark_30runs.json` reflect the
corrected name.


## Runtime and memory data

`runtime_memory_results.json` contains the raw timing and RAM measurements
behind Table 9 (Methods, Runtime and Memory Footprint): format detection,
dataset building, validation, and export timing plus RSS memory deltas,
measured on the confirmed current package (with `original_to_contiguous`
and `_merged_whole_dataset` fixes present) across all 6 datasets.
