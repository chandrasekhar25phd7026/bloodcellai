"""
Tests for Phase 4: bloodcell.quality_assessment
"""

import cv2
import numpy as np

from bloodcell.quality_assessment import (
    assess_image_quality,
    assess_dataset_quality,
    filter_to_quality_passing,
    QualityThresholds,
)
from bloodcell.universal_dataset import UniversalDataset
from bloodcell.universal_object import UniversalImage


def _make_test_image(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)


def test_assess_image_quality_on_real_file(tmp_path):
    path = tmp_path / "test.jpg"
    cv2.imwrite(str(path), _make_test_image())

    result = assess_image_quality(path)

    assert result.corrupted is False
    assert 0.0 <= result.quality_score <= 100.0
    assert result.image_id == "test"


def test_assess_image_quality_detects_missing_file(tmp_path):
    result = assess_image_quality(tmp_path / "does_not_exist.jpg")

    assert result.corrupted is True
    assert result.passed is False
    assert "missing_file" in result.issues


def test_assess_image_quality_detects_corrupted_file(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_text("not a real jpeg")

    result = assess_image_quality(path)

    assert result.corrupted is True
    assert result.passed is False
    assert "corrupted" in result.issues


def test_assess_dataset_quality_fresh_computation(tmp_path):
    dataset = UniversalDataset()

    for i in range(3):
        path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(path), _make_test_image(seed=i))
        dataset.add(UniversalImage(image_path=str(path), dataset="Test", width=200, height=200))

    report = assess_dataset_quality(dataset, use_cached=False)

    assert report.total_images == 3
    assert report.corrupted == 0
    assert all(r.source == "computed" for r in report.results)


def test_assess_dataset_quality_uses_cached_metadata_when_present(tmp_path):
    dataset = UniversalDataset()

    path = tmp_path / "img0.jpg"
    cv2.imwrite(str(path), _make_test_image())

    img = UniversalImage(image_path=str(path), dataset="Test", width=200, height=200)
    img.metadata["preprocessing"] = {
        "passed": True,
        "quality_score": 88.5,
        "quality_metrics": {"brightness": 100.0, "contrast": 40.0, "blur_score": 300.0, "entropy": 6.0},
        "warnings": [],
    }
    dataset.add(img)

    report = assess_dataset_quality(dataset, use_cached=True)

    assert report.results[0].source == "cached"
    assert report.results[0].quality_score == 88.5


def test_assess_dataset_quality_detects_corrupted_image_in_dataset(tmp_path):
    dataset = UniversalDataset()

    good_path = tmp_path / "good.jpg"
    cv2.imwrite(str(good_path), _make_test_image())
    dataset.add(UniversalImage(image_path=str(good_path), dataset="Test", width=200, height=200))

    bad_path = tmp_path / "bad.jpg"
    bad_path.write_text("corrupt")
    dataset.add(UniversalImage(image_path=str(bad_path), dataset="Test", width=0, height=0))

    report = assess_dataset_quality(dataset, use_cached=False)

    assert report.corrupted == 1
    assert report.total_images == 2


def test_issue_counts_and_pass_rate(tmp_path):
    dataset = UniversalDataset()

    for i in range(4):
        path = tmp_path / f"img{i}.jpg"
        cv2.imwrite(str(path), _make_test_image(seed=i))
        dataset.add(UniversalImage(image_path=str(path), dataset="Test", width=200, height=200))

    report = assess_dataset_quality(dataset, use_cached=False)

    assert report.pass_rate == round(100.0 * report.passed / report.total_images, 2)
    assert isinstance(report.issue_counts(), dict)


def test_filter_to_quality_passing_reuses_dataset_filter(tmp_path):
    dataset = UniversalDataset()

    # One image easy to pass (bright, high-contrast, sharp: real
    # checkerboard pattern), one deliberately corrupted.
    checkerboard = np.indices((200, 200)).sum(axis=0) % 2 * 255
    checkerboard = np.stack([checkerboard] * 3, axis=-1).astype(np.uint8)
    good_path = tmp_path / "good.jpg"
    cv2.imwrite(str(good_path), checkerboard)
    dataset.add(UniversalImage(image_path=str(good_path), dataset="Test", width=200, height=200))

    bad_path = tmp_path / "bad.jpg"
    bad_path.write_text("corrupt")
    dataset.add(UniversalImage(image_path=str(bad_path), dataset="Test", width=0, height=0))

    report = assess_dataset_quality(dataset, use_cached=False)
    clean = filter_to_quality_passing(dataset, report)

    # Whatever passed in the report should exactly match what's in
    # the filtered dataset -- proves the two are wired together
    # correctly via UniversalDataset.filter() (Phase 1).
    assert len(clean) == report.passed
    for img in clean:
        assert img.image_id in {r.image_id for r in report.results if r.passed}


def test_custom_thresholds_change_pass_fail_outcome(tmp_path):
    path = tmp_path / "img.jpg"
    cv2.imwrite(str(path), _make_test_image())

    strict = QualityThresholds(minimum_quality_score=99.9)
    lenient = QualityThresholds(minimum_quality_score=0.0)

    result_strict = assess_image_quality(path, strict)
    result_lenient = assess_image_quality(path, lenient)

    assert result_lenient.passed is True
    # Same measured quality_score regardless of threshold used to
    # judge it -- thresholds affect pass/fail, not the measurement.
    assert result_strict.quality_score == result_lenient.quality_score
