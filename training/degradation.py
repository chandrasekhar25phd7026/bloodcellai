"""
==============================================================
BloodCellAI Experiments — Controlled Dataset Degradation
==============================================================

File:
    degradation.py

Description
-----------
Produces controlled-degradation copies of a UniversalDataset for the
BDQI validation experiment: does a lower BDQI score actually predict
worse downstream training performance, or is it just a number with no
demonstrated relationship to anything that matters?

Each degradation level injects three kinds of real, controlled
problems at increasing severity into a fraction of images:

    - Annotation corruption: bounding boxes shifted/shrunk to
      near-zero area, or deleted outright.
    - Image degradation: heavy Gaussian blur and noise applied.
    - Label corruption: class labels randomly swapped to a wrong class.

The resulting dataset is otherwise a faithful, real copy of the
source images -- this is a controlled experiment, not a synthetic
toy dataset, so the resulting BDQI/training-performance relationship
is evidence about real degradation types the validation engine is
designed to catch.

Author:
    BloodCellAI Project
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class DegradationLevel:
    name: str
    fraction_affected: float   # 0.0-1.0, fraction of images touched
    annotation_corruption: bool = True
    image_degradation: bool = True
    label_corruption: bool = True
    use_darkening: bool = False
    """
    If True, image_degradation applies severe blur + darkening instead
    of blur + additive noise. Needed specifically for testing
    DatasetQualityGate's simple per-image quality_score (brightness/
    contrast/blur/entropy): confirmed empirically that additive noise,
    even at extreme severity, paradoxically INCREASES blur_score,
    contrast, and entropy simultaneously (random per-pixel noise mimics
    "sharp edges"/"high information content" to these specific metrics),
    so quality_score never drops no matter how much noise is added --
    it stays pinned at 100.0. Pure blur + darkening (underexposure)
    does not have this problem and genuinely lowers quality_score.
    Left False by default (existing blur+noise behavior, used by the
    BDQI validation experiment, which scores a different, more
    sophisticated validation-engine metric that correctly responds to
    annotation corruption regardless of this flag).
    """


# Five levels spanning "clean" to "heavily degraded" -- chosen to give
# a wide, evenly-spaced range for correlation analysis, not just two
# extremes.
DEGRADATION_LEVELS = [
    DegradationLevel("clean", 0.00),
    DegradationLevel("mild", 0.10),
    DegradationLevel("moderate", 0.25),
    DegradationLevel("severe", 0.50),
    DegradationLevel("extreme", 0.75),
]

# Dedicated level for testing DatasetQualityGate specifically (image
# quality only, no annotation/label corruption, using the
# blur+darkening recipe that quality_score can actually detect).
IMAGE_QUALITY_ONLY_SEVERE = DegradationLevel(
    "image_quality_only_severe",
    fraction_affected=0.5,
    annotation_corruption=False,
    image_degradation=True,
    label_corruption=False,
    use_darkening=True,
)


def _corrupt_annotation(obj, rng):
    """
    Corrupt one BoundingBox in place: either collapse it to
    near-zero area (mimicking the real zero-area defects found in
    BCCD during technical validation) or shift it partly off-image.
    """

    if rng.random() < 0.5:
        # Collapse to near-zero area -- the same defect class found
        # in real BCCD annotations during earlier validation.
        obj.w = 0.001
        obj.h = 0.001
    else:
        # Shift the box so it partially/fully leaves the image.
        obj.xc = min(1.2, obj.xc + rng.uniform(0.3, 0.6))
        obj.yc = min(1.2, obj.yc + rng.uniform(0.3, 0.6))

    return obj


def _degrade_image_file(src_path, dst_path, rng, use_darkening=False):
    """
    Write a heavily degraded copy of an image file to dst_path.

    Two recipes, chosen by `use_darkening`:

    - False (default): heavy blur + additive Gaussian noise. Used by
      the standard DEGRADATION_LEVELS (annotation + label corruption
      alongside this), validated against the deep validation-engine
      BDQI score, which correctly responds to this regardless of the
      image-level recipe used.

    - True: heavy blur + darkening (underexposure). Confirmed
      empirically necessary for testing DatasetQualityGate's simpler
      per-image quality_score (brightness/contrast/blur/entropy):
      additive noise paradoxically inflates blur_score/contrast/
      entropy simultaneously (random per-pixel noise mimics "sharp
      edges" to these metrics), so quality_score never actually drops
      no matter how much noise is added -- confirmed it stays pinned
      at 100.0 even at extreme noise levels. Blur + darkening does not
      have this problem.
    """

    image = cv2.imread(str(src_path))

    if image is None:
        return False

    if use_darkening:

        ksize = rng.choice([121, 151, 171])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        dark_factor = rng.uniform(0.05, 0.10)
        image = np.clip(image.astype(np.float32) * dark_factor, 0, 255).astype(np.uint8)

    else:

        # Heavy Gaussian blur
        ksize = rng.choice([9, 15, 21])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # Additive Gaussian noise
        noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(
            0, 35, image.shape
        )
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(str(dst_path), image)

    return True


def create_degraded_copy(
    source_dataset,
    level: DegradationLevel,
    output_image_dir,
    seed: int = 42,
    exclude_split: str = "test",
):
    """
    Create a degraded copy of a UniversalDataset at the given
    degradation level.

    Parameters
    ----------
    source_dataset : UniversalDataset
        The clean, real dataset to degrade (e.g., built BCCD). Must
        already have splits assigned (assign_splits()) before calling
        this, so `exclude_split` can be honored.

    level : DegradationLevel

    output_image_dir : str or Path
        Where degraded image files are written. Images not selected
        for degradation are copied through unchanged (still needed
        for a real, valid dataset directory).

    seed : int
        For reproducibility -- the same images/objects are selected
        for corruption at a given seed across repeated runs.

    exclude_split : str, default "test"
        Images in this split are NEVER degraded, regardless of
        `level`. This is essential for a valid experiment: the
        question is whether training on degraded data produces a
        worse model, evaluated against one constant, clean test set
        -- not whether degrading the test set itself lowers scores
        (which would be true trivially and tell us nothing about
        training-data quality).

    Returns
    -------
    UniversalDataset
        A new dataset object (deep copy of the source, with the
        selected fraction of eligible (non-excluded-split)
        images/annotations corrupted and image paths pointed at the
        new, possibly-degraded files).
    """

    import shutil

    output_image_dir = Path(output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    degraded = copy.deepcopy(source_dataset)
    degraded._invalidate_cache()

    eligible_indices = [
        i for i, image in enumerate(degraded.images)
        if getattr(image, "split", None) != exclude_split
    ]

    n_to_affect = int(round(len(eligible_indices) * level.fraction_affected))
    affected_indices = set(rng.sample(eligible_indices, n_to_affect))

    all_class_names = sorted(degraded.class_counts.keys())

    for i, image in enumerate(degraded.images):

        src_path = Path(image.image_path)
        dst_path = output_image_dir / src_path.name

        is_affected = i in affected_indices

        if is_affected and level.image_degradation:
            ok = _degrade_image_file(src_path, dst_path, rng, use_darkening=level.use_darkening)
            if not ok:
                shutil.copy2(src_path, dst_path)
        else:
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

        image.image_path = str(dst_path)

        if is_affected:

            if level.annotation_corruption:
                # Corrupt roughly half of this image's objects, not
                # all of them -- more realistic than every object in
                # an affected image being wrong.
                for obj in image.objects:
                    if rng.random() < 0.5:
                        _corrupt_annotation(obj, rng)

            if level.label_corruption and all_class_names:
                for obj in image.objects:
                    if rng.random() < 0.3:
                        wrong_choices = [
                            c for c in all_class_names if c != obj.class_name
                        ]
                        if wrong_choices:
                            obj.class_name = rng.choice(wrong_choices)

    # Rebuild class_counts to reflect post-corruption label swaps.
    degraded.class_counts.clear()
    for image in degraded.images:
        for obj in image.objects:
            degraded.class_counts[obj.class_name] += 1

    degraded.set_metadata("degradation_level", level.name)
    degraded.set_metadata("degradation_fraction", level.fraction_affected)

    return degraded
