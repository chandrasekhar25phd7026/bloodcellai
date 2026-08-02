"""
BloodCellAI Framework

File:
    universal_object.py

Description
-----------
Universal data structures for BloodCellAI.

These classes provide a common representation for object detection,
classification, segmentation, and future multimodal clinical analysis.

Version:
    1.1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# Universal Bounding Box / Annotation
# =============================================================================

@dataclass(slots=True)
class BoundingBox:
    """
    Universal annotation object.

    Supports:
        - YOLO
        - Pascal VOC
        - COCO
        - Faster R-CNN
        - DETR
        - Future segmentation
    """

    # -------------------------------------------------------------------------
    # Annotation Information
    # -------------------------------------------------------------------------

    annotation_id: str = ""

    class_id: int = -1

    class_name: str = ""

    # -------------------------------------------------------------------------
    # YOLO Coordinates (Normalized)
    # -------------------------------------------------------------------------

    xc: float = 0.0

    yc: float = 0.0

    w: float = 0.0

    h: float = 0.0

    # -------------------------------------------------------------------------
    # Prediction Information
    # -------------------------------------------------------------------------

    confidence: Optional[float] = None

    # -------------------------------------------------------------------------
    # COCO Compatibility
    # -------------------------------------------------------------------------

    area: float = 0.0

    iscrowd: bool = False

    # -------------------------------------------------------------------------
    # Source Information
    # -------------------------------------------------------------------------

    source: str = ""

    # -------------------------------------------------------------------------
    # Additional Metadata
    # -------------------------------------------------------------------------

    metadata: Dict = field(default_factory=dict)


# =============================================================================
# Universal Image Record
# =============================================================================

@dataclass(slots=True)
class UniversalImage:
    """
    Universal image representation.
    """

    image_path: str

    dataset: str

    width: int

    height: int

    split: str = "train"

    # Stable per-image identifier. Auto-derived from the filename stem in
    # __post_init__ when not explicitly provided, so every UniversalImage
    # has one without every adapter needing to set it by hand.
    image_id: str = ""

    metadata: Dict = field(default_factory=dict)

    objects: List[BoundingBox] = field(default_factory=list)

    def __post_init__(self):

        if not self.image_id and self.image_path:
            self.image_id = Path(self.image_path).stem


# =============================================================================
# Clinical Record
# =============================================================================

@dataclass(slots=True)
class ClinicalRecord:
    """
    Clinical information associated with an image.
    """

    image_path: str

    dataset: str

    cbc: Dict = field(default_factory=dict)

    morphology: List[str] = field(default_factory=list)

    metadata: Dict = field(default_factory=dict)