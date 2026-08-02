"""
Tests for:
- bloodcell.file_matcher.FileMatcher, specifically the cross-dataset
  annotation contamination bug found while wiring up RBCMorphology.
- bloodcell.adapters.ClinicalAdapter / parsers.parse_cbc_report /
  parsers.extract_morphology_findings, which existed as dead/undefined
  code before this pass.
"""

import pathlib

from bloodcell.file_matcher import FileMatcher
from bloodcell.adapters import ClinicalAdapter
from bloodcell.parsers import parse_cbc_report, extract_morphology_findings


def test_file_matcher_does_not_cross_contaminate_datasets(tmp_path: pathlib.Path):
    # Two separate "dataset" folders that happen to share a filename
    # stem -- exactly the scenario that silently broke before this fix,
    # since find_annotation() used to recompute its own (too-broad)
    # scope instead of trusting the index build_index() already built.
    dataset_a = tmp_path / "datasets" / "DatasetA"
    dataset_b = tmp_path / "datasets" / "DatasetB"
    dataset_a.mkdir(parents=True)
    dataset_b.mkdir(parents=True)

    (dataset_a / "sample1.txt").write_text("this belongs to A")
    (dataset_b / "sample1.txt").write_text("this belongs to B")

    matcher_a = FileMatcher()
    matcher_a.build_index(dataset_a)

    matcher_b = FileMatcher()
    matcher_b.build_index(dataset_b)

    found_a = matcher_a.find_annotation(dataset_a / "sample1.jpg")
    found_b = matcher_b.find_annotation(dataset_b / "sample1.jpg")

    assert found_a is not None
    assert found_b is not None
    assert found_a.read_text() == "this belongs to A"
    assert found_b.read_text() == "this belongs to B"
    assert found_a != found_b


def test_parse_cbc_report_extracts_numeric_fields(tmp_path: pathlib.Path):
    report = tmp_path / "report.txt"
    report.write_text(
        "Hemoglobin: 13.5\n"
        "RBC_Count: 4.8\n"
        "WBC_Count: 7200\n"
        "Morphology: Microcytic, Hypochromic\n"
    )

    cbc = parse_cbc_report(report)

    assert cbc["Hemoglobin"] == 13.5
    assert cbc["RBC_Count"] == 4.8
    assert cbc["WBC_Count"] == 7200
    # The Morphology line is not a numeric CBC field and must not leak in.
    assert "Morphology" not in cbc


def test_extract_morphology_findings_parses_finding_list(tmp_path: pathlib.Path):
    report = tmp_path / "report.txt"
    report.write_text(
        "Hemoglobin: 13.5\n"
        "Morphology: Microcytic, Hypochromic\n"
    )

    findings = extract_morphology_findings(report)

    assert findings == ["Microcytic", "Hypochromic"]


def test_extract_morphology_findings_empty_when_no_morphology_line(tmp_path: pathlib.Path):
    report = tmp_path / "report.txt"
    report.write_text("Hemoglobin: 13.5\n")

    assert extract_morphology_findings(report) == []


def test_clinical_adapter_matches_pipeline_interface(tmp_path: pathlib.Path):
    # Before this pass, ClinicalAdapter.convert() required two separate
    # files (morphology_file, cbc_file) as positional args and called
    # extract_morphology_findings()/parse_cbc_report(), neither of which
    # was defined anywhere -- it could never run, even in isolation.
    report = tmp_path / "report.txt"
    report.write_text(
        "Hemoglobin: 12.0\n"
        "Morphology: Elliptocyte\n"
    )

    adapter = ClinicalAdapter()
    record = adapter.convert(
        annotation_file=report,
        image_path=str(tmp_path / "sample.jpg"),
        dataset="Clinical",
    )

    assert record.cbc["Hemoglobin"] == 12.0
    assert record.morphology == ["Elliptocyte"]
