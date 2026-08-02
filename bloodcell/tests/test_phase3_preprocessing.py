"""
Tests for Phase 3: preprocessing pipeline fixes and its integration
with dataset loading (bloodcell.pipeline / bloodcell.universal_builder).

Run from the project root with both `bloodcell` and `preprocessing` /
`transforms` importable as top-level sibling packages.
"""

import numpy as np
import pytest

from preprocessing.preprocessing_config import PreprocessingConfig
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from preprocessing.preprocessing_manager import PreprocessingManager
from preprocessing.preprocessing_models import ImageQualityMetrics


def _make_test_image(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Core pipeline correctness (previously totally broken/untested)
# ---------------------------------------------------------------------------

def test_pipeline_builds_and_runs_without_crashing():
    config = PreprocessingConfig()
    pipeline = PreprocessingPipeline(config)

    # _build_pipeline() used to be missing its return statement and
    # most of its transform appends -- self._pipeline was silently
    # None, guaranteeing a crash. If that regressed, this would fail
    # immediately on construction or on transform_count access.
    assert pipeline.transform_count > 0

    image = _make_test_image()
    result = pipeline.preprocess(image)

    assert result is not None
    assert result.processed_image is not None


def test_processed_image_property_not_broken():
    """
    PreprocessingPipeline.processed_image used to reference a
    nonexistent `.image` attribute on PreprocessingResult (it's
    `.processed_image`) -- would have raised AttributeError.
    """

    config = PreprocessingConfig()
    pipeline = PreprocessingPipeline(config)
    pipeline.preprocess(_make_test_image())

    assert pipeline.processed_image is not None
    assert pipeline.processed_image.shape[2] == 3


def test_quality_metrics_are_actually_propagated():
    """
    Compose.__call__ used to never copy a transform's computed
    quality metrics into the final PreprocessingResult -- it silently
    stayed at all-zero defaults forever, even on a real image.
    """

    config = PreprocessingConfig()
    pipeline = PreprocessingPipeline(config)

    result = pipeline.preprocess(_make_test_image())

    metrics = result.quality_metrics
    # A real random image should not produce every metric at exactly
    # zero -- that all-zero signature is exactly what the propagation
    # bug looked like.
    assert not (
        metrics.brightness == 0.0
        and metrics.contrast == 0.0
        and metrics.entropy == 0.0
        and metrics.quality_score == 0.0
    )


def test_quality_score_is_computed_within_expected_range():
    """
    _quality_score() was called by apply() but never defined anywhere
    -- guaranteed AttributeError on any real use.
    """

    config = PreprocessingConfig()
    pipeline = PreprocessingPipeline(config)
    result = pipeline.preprocess(_make_test_image())

    assert 0.0 <= result.quality_metrics.quality_score <= 100.0


def test_image_quality_metrics_passed_is_settable():
    """
    ImageQualityMetrics.passed used to be a read-only @property
    (quality_score >= 70.0), but quality.py's _evaluate_quality()
    assigns to it directly -- that would raise AttributeError.
    """

    metrics = ImageQualityMetrics()
    metrics.passed = False  # must not raise
    assert metrics.passed is False


def test_manager_preprocesses_a_real_file_path():
    """
    End-to-end smoke test of the whole stack via a real temp file,
    exercising resize/quality/normalize together.
    """

    import cv2
    import tempfile
    import os

    config = PreprocessingConfig()
    manager = PreprocessingManager(config)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.jpg")
        cv2.imwrite(path, _make_test_image())

        result = manager.preprocess_image(path)

        assert result.processed_image is not None
        assert result.quality_metrics.quality_score >= 0.0


# ---------------------------------------------------------------------------
# Dataset-loading integration (the actual Phase 3 goal)
# ---------------------------------------------------------------------------

def test_build_from_info_attaches_preprocessing_metadata(tmp_path):
    """
    Enabling preprocessing on UniversalBuilder should attach a real
    "preprocessing" metadata block to every built image -- this is
    the actual "automatically validated and preprocessed during
    dataset creation" integration goal.
    """

    import cv2
    from bloodcell.dataset_builder import UniversalDatasetBuilder
    from bloodcell.universal_builder import UniversalBuilder

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"img{i}.jpg"), _make_test_image(seed=i))
        (tmp_path / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    db = UniversalDatasetBuilder()
    info = db.auto_register(tmp_path, dataset_name="preprocessing_integration_test")
    db.prepare("preprocessing_integration_test")

    ub = UniversalBuilder(tmp_path)
    ub.enable_preprocessing()

    dataset, built = ub.build_from_info(info)

    assert built == 3

    for img in dataset.images:
        assert "preprocessing" in img.metadata
        assert "quality_score" in img.metadata["preprocessing"]


def test_preprocessing_disabled_by_default(tmp_path):
    """
    Backward compatibility: without calling enable_preprocessing(),
    no preprocessing metadata should appear at all.
    """

    import cv2
    from bloodcell.dataset_builder import UniversalDatasetBuilder
    from bloodcell.universal_builder import UniversalBuilder

    cv2.imwrite(str(tmp_path / "img0.jpg"), _make_test_image())
    (tmp_path / "img0.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    db = UniversalDatasetBuilder()
    info = db.auto_register(tmp_path, dataset_name="preprocessing_off_test")
    db.prepare("preprocessing_off_test")

    ub = UniversalBuilder(tmp_path)
    # enable_preprocessing() NOT called

    dataset, built = ub.build_from_info(info)

    assert built == 1
    assert "preprocessing" not in dataset.images[0].metadata


def test_corrupted_image_is_flagged_not_dropped(tmp_path):
    """
    A corrupt/unreadable image should stay in the dataset with a
    flagged preprocessing failure, not silently disappear or crash
    the whole build.
    """

    import cv2
    from bloodcell.dataset_builder import UniversalDatasetBuilder
    from bloodcell.universal_builder import UniversalBuilder

    cv2.imwrite(str(tmp_path / "good.jpg"), _make_test_image())
    (tmp_path / "good.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    (tmp_path / "bad.jpg").write_text("not a real jpeg")
    (tmp_path / "bad.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    db = UniversalDatasetBuilder()
    info = db.auto_register(tmp_path, dataset_name="preprocessing_corrupt_test")
    db.prepare("preprocessing_corrupt_test")

    ub = UniversalBuilder(tmp_path)
    ub.enable_preprocessing()

    dataset, built = ub.build_from_info(info)

    assert built == 2  # both images still built (annotation parsing doesn't fail)

    by_id = {img.image_id: img for img in dataset.images}

    assert by_id["good"].metadata["preprocessing"]["passed"] is True
    assert by_id["bad"].metadata["preprocessing"]["passed"] is False
    assert "error" in by_id["bad"].metadata["preprocessing"]
