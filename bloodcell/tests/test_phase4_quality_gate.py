"""
Tests for Phase 4: bloodcell.quality_gate.DatasetQualityGate.
"""

import cv2
import numpy as np


def _make_good_image():
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)


def _make_flat_low_quality_image():
    return np.full((200, 200, 3), 128, dtype=np.uint8)


def test_assess_detects_corrupted_file(tmp_path):
    from bloodcell.dataset_builder import UniversalDatasetBuilder
    from bloodcell.universal_builder import UniversalBuilder
    from bloodcell.quality_gate import DatasetQualityGate

    cv2.imwrite(str(tmp_path / "good.jpg"), _make_good_image())
    (tmp_path / "good.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    (tmp_path / "bad.jpg").write_text("not a real jpeg")
    (tmp_path / "bad.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    db = UniversalDatasetBuilder()
    info = db.auto_register(tmp_path, dataset_name="qgate_corrupt_test")
    db.prepare("qgate_corrupt_test")

    ub = UniversalBuilder(tmp_path)
    ub.enable_preprocessing()
    dataset, built = ub.build_from_info(info)

    gate = DatasetQualityGate()
    report = gate.assess(dataset)

    assert report.total_images == 2
    assert report.corrupted_count == 1
    assert report.passed_count == 1


def test_filter_passing_excludes_corrupted_and_low_quality(tmp_path):
    from bloodcell.dataset_builder import UniversalDatasetBuilder
    from bloodcell.universal_builder import UniversalBuilder
    from bloodcell.quality_gate import DatasetQualityGate

    cv2.imwrite(str(tmp_path / "good.jpg"), _make_good_image())
    (tmp_path / "good.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    cv2.imwrite(str(tmp_path / "flat.jpg"), _make_flat_low_quality_image())
    (tmp_path / "flat.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    db = UniversalDatasetBuilder()
    info = db.auto_register(tmp_path, dataset_name="qgate_filter_test")
    db.prepare("qgate_filter_test")

    ub = UniversalBuilder(tmp_path)
    ub.enable_preprocessing()
    dataset, built = ub.build_from_info(info)

    gate = DatasetQualityGate(minimum_quality_score=50.0)
    filtered, report = gate.filter_passing(dataset)

    assert len(filtered) < len(dataset)
    assert all(
        img.image_id != "flat" for img in filtered
    )


def test_on_demand_computation_when_no_preprocessing_metadata(tmp_path):
    """
    A dataset built WITHOUT UniversalBuilder.enable_preprocessing()
    has no metadata["preprocessing"] block yet -- the gate must
    compute quality metrics itself rather than erroring out.
    """

    from bloodcell.dataset_builder import UniversalDatasetBuilder
    from bloodcell.universal_builder import UniversalBuilder
    from bloodcell.quality_gate import DatasetQualityGate

    cv2.imwrite(str(tmp_path / "good.jpg"), _make_good_image())
    (tmp_path / "good.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    db = UniversalDatasetBuilder()
    info = db.auto_register(tmp_path, dataset_name="qgate_ondemand_test2")
    db.prepare("qgate_ondemand_test2")

    ub = UniversalBuilder(tmp_path)
    # enable_preprocessing() deliberately NOT called
    dataset, built = ub.build_from_info(info)

    assert "preprocessing" not in dataset.images[0].metadata

    gate = DatasetQualityGate()
    report = gate.assess(dataset)

    assert report.total_images == 1
    assert report.records[0].quality_score is not None


def test_report_pass_rate_and_summary_text():
    from bloodcell.quality_gate import DatasetQualityReport

    report = DatasetQualityReport(
        total_images=4, passed_count=3, failed_count=1, corrupted_count=1
    )

    assert report.pass_rate == 0.75
    assert "Dataset Quality Report" in report.summary_text()
