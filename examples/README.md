# Examples

Two standalone, runnable examples demonstrating the core BloodCellAI workflow.

## `example_build_dataset.py`

Detects a dataset's format, builds it into BloodCellAI's harmonized
representation, filters it with the quality gate, and exports it in a
standard training-ready layout.

```bash
python example_build_dataset.py /path/to/your/dataset /path/to/output
```

## `example_validation.py`

Runs the validation engine on a dataset and prints its full Blood Dataset
Quality Index (BDQI) report, including component scores and any issues found.

```bash
python example_validation.py /path/to/your/dataset
```

## Requirements

Both examples require BloodCellAI's `bloodcell` package (and `validation`
for the second example) to be installed -- see the main [README](../README.md)
for installation instructions. Neither example requires a GPU; both run
in seconds to tens of seconds depending on dataset size (see Table 9 in
the accompanying paper for measured timing on six real datasets).

## Trying these on a real dataset

Any of the six datasets used in this study's benchmark (Table 1 in the
paper) will work directly with `example_build_dataset.py`. Datasets using
point annotations or whole-dataset JSON formats (like two of the six used
in this study) need manual registration first -- see the comment at the
bottom of `example_build_dataset.py` for how.
