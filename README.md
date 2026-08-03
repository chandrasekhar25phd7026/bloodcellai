# BloodCellAI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21772853.svg)](https://doi.org/10.5281/zenodo.21772853)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/chandrasekhar25phd7026/bloodcellai)](https://github.com/chandrasekhar25phd7026/bloodcellai/releases)
[![Tests](https://img.shields.io/badge/tests-76%20passing-brightgreen.svg)](bloodcell/tests/)

An automated format-detection, quality-assessment, and harmonization framework for multi-source blood cell imaging datasets.

BloodCellAI looks at a blood cell dataset's raw folder contents, automatically works out its annotation format (Pascal VOC, YOLO, COCO, point annotations, whole-dataset JSON, or folder-per-class), converts it into one shared internal representation, checks its quality with a validated composite score (the Blood Dataset Quality Index, BDQI), and exports it ready for training — for both object detection and whole-image classification tasks.

## Why

Blood cell image analysis with deep learning is fragmented across dozens of independent, single-dataset studies, each using a different annotation convention. BloodCellAI harmonizes datasets automatically so they can be compared, combined, and benchmarked under one consistent pipeline.

Validated on real, public datasets — not just described:
- **BDQI correctly tracks downstream training performance** (Spearman ρ = 1.00) on two structurally different datasets under controlled degradation.
- **4 of 6 real public datasets auto-detected correctly** with no manual setup.
- **Benchmarked across 6 datasets × 5 seeds (30 training runs)**, reporting mean ± SD rather than single-run point estimates.
- **76 automated tests**, all passing against both synthetic fixtures and real datasets.

See the accompanying paper for full methodology, benchmark results, and two honestly-reported ablation studies (including one correction to our own original single-run finding).

## Installation

```bash
git clone https://github.com/chandrasekhar25phd7026/bloodcellai.git
cd bloodcellai
pip install -e ./bloodcell
pip install -e ./preprocessing
pip install -e ./transforms
pip install -e ./training
```

Each package can also be installed independently if you only need part of the pipeline.

## Quickstart

```python
from pathlib import Path
from bloodcell.dataset_builder import UniversalDatasetBuilder
from bloodcell.universal_builder import UniversalBuilder
from bloodcell.quality_gate import DatasetQualityGate
from bloodcell.dataset_export import export_yolo

# Point at any dataset folder -- format is detected automatically
builder = UniversalDatasetBuilder()
info = builder.auto_register("/path/to/your/dataset")
builder.prepare(info.name)

# Build, with optional preprocessing/quality scoring
ub = UniversalBuilder("/path/to/working/dir")
ub.enable_preprocessing()
dataset, built = ub.build_from_info(info)

# Filter out low-quality images before training
gate = DatasetQualityGate(minimum_quality_score=50.0)
clean_dataset, report = gate.filter_passing(dataset)

# Split and export in a standard, training-ready layout
clean_dataset.assign_splits(train=0.8, val=0.1, test=0.1)
export_yolo(clean_dataset, "/path/to/export")
```

If your dataset's format isn't auto-detected, register it manually with a `DatasetInfo` object (task, annotation type, class mapping) — everything downstream works identically either way.

## Architecture

See `docs/architecture_diagram.png` for the full pipeline, and `docs/module_diagram.png` for the package/module structure.

## Running the tests

```bash
pytest bloodcell/tests/
```

76 tests, covering synthetic fixtures and real public datasets.

## Reproducing the paper's results

See `supplementary/` for the complete raw results (all 30 benchmark runs, 25 ablation runs, BDQI validation data, runtime/memory measurements) and standalone reproducibility scripts for every experiment reported in the paper.

## Citation

If you use this framework, please cite:

```bibtex
@software{bloodcellai2026,
  title = {BloodCellAI: An Automated Format-Detection, Quality-Assessment, and Harmonization Framework for Multi-Source Blood Cell Imaging Datasets},
  author = {Muttangi, Chandrasekhar},
  year = {2026},
  url = {https://github.com/chandrasekhar25phd7026/bloodcellai},
  doi = {10.5281/zenodo.21772853}
}
```

A `CITATION.cff` file is also included, so GitHub's "Cite this repository" button will generate this automatically.

## License

MIT License — see [LICENSE](LICENSE).

## Data availability

BloodCellAI does not redistribute the third-party datasets used in its validation. Each is publicly available from its original source; see the accompanying paper's References and Data Records sections for citations and access links.
