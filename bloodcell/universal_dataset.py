"""
==============================================================
BloodCellAI Enterprise Universal Dataset
==============================================================

File:
    universal_dataset.py

Description
-----------
The central in-memory container for a harmonized BloodCellAI dataset.
Holds a flat list of UniversalImage (or ClinicalRecord) entries built
from one or more source datasets, plus:

    - Indexing        fast lookup by image_id / path / dataset / split / class
    - Searching       predicate-based search across all images
    - Filtering       return a new UniversalDataset matching a query
    - Metadata        dataset-level provenance (created_at, version, tags, ...)
    - Statistics      dimension/class/split distributions, computed lazily
    - Caching         indexes and statistics are built once and reused
                       until the dataset actually changes (add()/filter())
    - Train/Val/Test  split assignment (random or stratified) and access

Design notes
------------
This class intentionally stays dependency-light (no import of the
`validation` package) so that building and querying a dataset never
requires pulling in the full validation/BDQI engine. `statistics()`
here is a fast, self-contained summary; `validation.DatasetValidator`
/ `DatasetStatisticsEngine` remain the place for deeper quality
scoring (BDQI) built on top of what's here.

All existing public surface (fields: images, metadata, warnings,
dataset_counts, class_counts; methods: add(), __len__, summary(),
export_pickle(), export_json()) is preserved unchanged so nothing
that already depends on UniversalDataset breaks.

Version:
    2.0.0
"""

from __future__ import annotations

import json
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional


@dataclass
class UniversalDataset:

    images: list = field(default_factory=list)

    metadata: dict = field(default_factory=dict)

    warnings: list = field(default_factory=list)

    dataset_counts: Counter = field(default_factory=Counter)

    class_counts: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:

        # Cache / index state. None of this is a dataclass field --
        # it's rebuilt lazily from `images`, never serialized, and
        # never expected to be passed to the constructor.
        self._index_dirty: bool = True

        self._index_by_id: Dict[str, int] = {}
        self._index_by_path: Dict[str, int] = {}
        self._index_by_dataset: Dict[str, List[int]] = defaultdict(list)
        self._index_by_split: Dict[str, List[int]] = defaultdict(list)
        self._index_by_class: Dict[str, List[int]] = defaultdict(list)

        self._statistics_cache: Optional[dict] = None

        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()

        self.metadata.setdefault("version", 1)
        self.metadata.setdefault("tags", [])

    # =========================================================================
    # Core mutation (existing behavior, extended to invalidate caches)
    # =========================================================================

    def add(self, image) -> None:

        self.images.append(image)

        self.dataset_counts[image.dataset] += 1

        for obj in getattr(image, "objects", []):

            self.class_counts[obj.class_name] += 1

        self._invalidate_cache()

    def __len__(self) -> int:

        return len(self.images)

    def __iter__(self) -> Iterator:

        return iter(self.images)

    def __getitem__(self, index):

        return self.images[index]

    def __contains__(self, image_id: str) -> bool:

        self._ensure_index()

        return image_id in self._index_by_id

    def __repr__(self) -> str:

        return (
            f"UniversalDataset(images={len(self.images)}, "
            f"datasets={dict(self.dataset_counts)}, "
            f"classes={len(self.class_counts)})"
        )

    # =========================================================================
    # Caching / Indexing
    # =========================================================================

    def _invalidate_cache(self) -> None:
        """
        Mark cached indexes and statistics as stale. Called whenever
        the dataset's contents change (add(), assign_splits(), ...).

        Indexes and statistics are NOT rebuilt here -- rebuilding on
        every single add() would make building an N-image dataset via
        repeated add() calls O(N^2). Instead this just flips a flag,
        and the next call to any indexed/statistics method rebuilds
        once, lazily.
        """

        self._index_dirty = True
        self._statistics_cache = None

    def _ensure_index(self) -> None:
        """
        Rebuild all indexes if they're stale. Cheap no-op otherwise.
        """

        if not self._index_dirty:
            return

        self._index_by_id = {}
        self._index_by_path = {}
        self._index_by_dataset = defaultdict(list)
        self._index_by_split = defaultdict(list)
        self._index_by_class = defaultdict(list)

        for i, image in enumerate(self.images):

            image_id = getattr(image, "image_id", None)
            if image_id:
                self._index_by_id[image_id] = i

            image_path = getattr(image, "image_path", None)
            if image_path:
                self._index_by_path[str(image_path)] = i

            dataset_name = getattr(image, "dataset", None)
            if dataset_name:
                self._index_by_dataset[dataset_name].append(i)

            split = getattr(image, "split", None)
            if split:
                self._index_by_split[split].append(i)

            for obj in getattr(image, "objects", []):
                class_name = getattr(obj, "class_name", None)
                if class_name:
                    self._index_by_class[class_name].append(i)

        self._index_dirty = False

    # =========================================================================
    # Indexing / Lookup
    # =========================================================================

    def get_by_id(self, image_id: str):
        """
        O(1) lookup by image_id. Returns None if not found.
        """

        self._ensure_index()

        index = self._index_by_id.get(image_id)

        return self.images[index] if index is not None else None

    def get_by_path(self, image_path: str):
        """
        O(1) lookup by image_path. Returns None if not found.
        """

        self._ensure_index()

        index = self._index_by_path.get(str(image_path))

        return self.images[index] if index is not None else None

    def image_ids(self) -> List[str]:
        """
        All known image_ids, in dataset order.
        """

        self._ensure_index()

        return list(self._index_by_id.keys())

    # =========================================================================
    # Searching
    # =========================================================================

    def find(self, predicate: Callable[[Any], bool]) -> List[Any]:
        """
        Return every image matching an arbitrary predicate.

        Example
        -------
        wide_images = dataset.find(lambda img: img.width > 1024)
        """

        return [image for image in self.images if predicate(image)]

    def find_one(self, predicate: Callable[[Any], bool]):
        """
        Return the first image matching a predicate, or None.
        """

        for image in self.images:
            if predicate(image):
                return image

        return None

    # =========================================================================
    # Filtering (returns a NEW UniversalDataset)
    # =========================================================================

    def filter(self, predicate: Callable[[Any], bool]) -> "UniversalDataset":
        """
        Return a new UniversalDataset containing only images matching
        `predicate`. The original dataset is left untouched.
        """

        subset = UniversalDataset()

        subset.metadata = dict(self.metadata)
        subset.metadata["filtered_from"] = self.metadata.get("dataset_id", id(self))
        subset.metadata["filter_created_at"] = datetime.now(timezone.utc).isoformat()

        for image in self.images:
            if predicate(image):
                subset.add(image)

        return subset

    def by_dataset(self, name: str) -> "UniversalDataset":
        """
        Subset containing only images from one source dataset.
        """

        return self.filter(lambda img: getattr(img, "dataset", None) == name)

    def by_split(self, split: str) -> "UniversalDataset":
        """
        Subset containing only images assigned to one split
        ("train" / "val" / "test", or any custom split name).
        """

        return self.filter(lambda img: getattr(img, "split", None) == split)

    def by_class(self, class_name: str) -> "UniversalDataset":
        """
        Subset containing only images with at least one object of
        the given class.
        """

        return self.filter(
            lambda img: any(
                getattr(obj, "class_name", None) == class_name
                for obj in getattr(img, "objects", [])
            )
        )

    def search(
        self,
        *,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        class_name: Optional[str] = None,
        min_objects: Optional[int] = None,
        max_objects: Optional[int] = None,
        min_width: Optional[int] = None,
        min_height: Optional[int] = None,
    ) -> "UniversalDataset":
        """
        Composable query across multiple criteria at once (all
        supplied criteria must match -- logical AND). Any criterion
        left as None is not applied.

        Example
        -------
        small_bccd_train = dataset.search(
            dataset="BCCD", split="train", max_objects=2
        )
        """

        def predicate(img) -> bool:

            if dataset is not None and getattr(img, "dataset", None) != dataset:
                return False

            if split is not None and getattr(img, "split", None) != split:
                return False

            objects = getattr(img, "objects", [])

            if class_name is not None:
                if not any(getattr(o, "class_name", None) == class_name for o in objects):
                    return False

            n_objects = len(objects)

            if min_objects is not None and n_objects < min_objects:
                return False

            if max_objects is not None and n_objects > max_objects:
                return False

            if min_width is not None and getattr(img, "width", 0) < min_width:
                return False

            if min_height is not None and getattr(img, "height", 0) < min_height:
                return False

            return True

        return self.filter(predicate)

    # =========================================================================
    # Metadata
    # =========================================================================

    def set_metadata(self, key: str, value: Any) -> None:

        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:

        return self.metadata.get(key, default)

    def update_metadata(self, **kwargs) -> None:

        self.metadata.update(kwargs)

    def add_tag(self, tag: str) -> None:

        tags = self.metadata.setdefault("tags", [])

        if tag not in tags:
            tags.append(tag)

    # =========================================================================
    # Statistics
    # =========================================================================

    def statistics(self, recompute: bool = False) -> dict:
        """
        Compute (and cache) a fast, self-contained statistics summary.

        This is intentionally lightweight and dependency-free -- for
        deeper quality scoring (BDQI, annotation/image/class/bbox
        validity issues), use validation.DatasetValidator on top of
        this dataset instead. This method answers "what does this
        dataset look like" quickly; that engine answers "how good is
        it and why."

        Cached after first call; recomputed automatically the next
        time it's called after add()/filter() change the dataset, or
        immediately if `recompute=True`.
        """

        if self._statistics_cache is not None and not recompute:
            return self._statistics_cache

        widths = [getattr(img, "width", None) for img in self.images]
        heights = [getattr(img, "height", None) for img in self.images]
        widths = [w for w in widths if w]
        heights = [h for h in heights if h]

        objects_per_image = [
            len(getattr(img, "objects", [])) for img in self.images
        ]

        total_objects = sum(objects_per_image)

        stats: dict = {
            "total_images": len(self.images),
            "total_objects": total_objects,
            "number_of_classes": len(self.class_counts),
            "dataset_counts": dict(self.dataset_counts),
            "class_counts": dict(self.class_counts),
            "split_counts": self.split_counts(),
            "objects_per_image": {
                "mean": _safe_mean(objects_per_image),
                "min": min(objects_per_image) if objects_per_image else 0,
                "max": max(objects_per_image) if objects_per_image else 0,
            },
            "image_width": {
                "mean": _safe_mean(widths),
                "min": min(widths) if widths else None,
                "max": max(widths) if widths else None,
            },
            "image_height": {
                "mean": _safe_mean(heights),
                "min": min(heights) if heights else None,
                "max": max(heights) if heights else None,
            },
            "class_balance": self.class_balance(),
        }

        self._statistics_cache = stats

        return stats

    def class_balance(self) -> Dict[str, float]:
        """
        Fraction of total object count contributed by each class.
        Useful for spotting severe class imbalance at a glance.
        """

        total = sum(self.class_counts.values())

        if total == 0:
            return {}

        return {
            class_name: round(count / total, 4)
            for class_name, count in self.class_counts.items()
        }

    def split_counts(self) -> Dict[str, int]:
        """
        Number of images per split (train/val/test/... or
        "unassigned" for images that haven't been assigned a split).
        """

        self._ensure_index()

        counts = {
            split: len(indices)
            for split, indices in self._index_by_split.items()
        }

        unassigned = len(self.images) - sum(counts.values())

        if unassigned:
            counts["unassigned"] = unassigned

        return counts

    def summary(self) -> dict:
        """
        Existing lightweight summary, kept unchanged for backward
        compatibility with anything already calling it.
        """

        return {
            "Images": len(self.images),
            "Datasets": dict(self.dataset_counts),
            "Classes": dict(self.class_counts),
            "Warnings": len(self.warnings),
        }

    # =========================================================================
    # Train / Validation / Test Support
    # =========================================================================

    def assign_splits(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        stratify_by: Optional[str] = "dataset",
        seed: int = 42,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """
        Assign a train/val/test split to every image's `.split`
        field.

        Parameters
        ----------
        train, val, test
            Proportions (must sum to ~1.0).

        stratify_by
            "dataset" (default) -- split each source dataset's images
            independently, so the train/val/test ratio holds within
            every source dataset, not just overall (important once
            you're combining several datasets of very different
            sizes). Pass None for a single global random split
            instead.

        seed
            Random seed, for reproducible splits.

        overwrite
            If False (default), images that already have a non-empty,
            non-"train"-default `.split` explicitly set are left
            alone. Set True to reassign everything.

        Returns
        -------
        dict
            Resulting split_counts(), for convenience.
        """

        total = train + val + test

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train + val + test must sum to 1.0, got {total}"
            )

        rng = random.Random(seed)

        if stratify_by == "dataset":
            groups: Dict[str, List[int]] = defaultdict(list)
            for i, image in enumerate(self.images):
                groups[getattr(image, "dataset", "")].append(i)
        else:
            groups = {"__all__": list(range(len(self.images)))}

        for _, indices in groups.items():

            indices = list(indices)
            rng.shuffle(indices)

            n = len(indices)
            n_train = round(n * train)
            n_val = round(n * val)

            train_idx = set(indices[:n_train])
            val_idx = set(indices[n_train:n_train + n_val])
            # everything else -> test, avoids rounding gaps/overlaps

            for pos in indices:

                image = self.images[pos]

                if not overwrite and getattr(image, "split", None) not in (None, "", "train"):
                    continue

                if pos in train_idx:
                    image.split = "train"
                elif pos in val_idx:
                    image.split = "val"
                else:
                    image.split = "test"

        self._invalidate_cache()

        return self.split_counts()

    def train_set(self) -> "UniversalDataset":
        return self.by_split("train")

    def val_set(self) -> "UniversalDataset":
        return self.by_split("val")

    def test_set(self) -> "UniversalDataset":
        return self.by_split("test")

    # =========================================================================
    # Export (existing behavior, preserved)
    # =========================================================================

    def export_pickle(self, file) -> None:

        with open(file, "wb") as f:
            pickle.dump(self, f)

    def export_json(self, file) -> None:

        with open(file, "w") as f:
            json.dump(self.summary(), f, indent=4)

    def export_statistics_json(self, file) -> None:
        """
        Export the richer statistics() output (not just summary())
        as JSON -- useful for feeding a report generator or a paper's
        data-descriptor table without recomputing stats separately.
        """

        with open(file, "w") as f:
            json.dump(self.statistics(), f, indent=4, default=str)


def _safe_mean(values: Iterable) -> Optional[float]:

    values = list(values)

    if not values:
        return None

    return round(sum(values) / len(values), 3)
