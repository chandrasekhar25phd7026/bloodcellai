"""
Tests for UniversalDataset's enhanced capabilities: indexing,
searching, filtering, metadata, statistics, caching, and
train/val/test support.
"""

from bloodcell.universal_dataset import UniversalDataset
from bloodcell.universal_object import UniversalImage, BoundingBox


def _make_dataset():
    ds = UniversalDataset()

    specs = [
        ("BCCD", 640, 480, ["RBC", "WBC"]),
        ("BCCD", 640, 480, ["RBC"]),
        ("BCCD", 800, 600, ["RBC", "RBC", "Platelet"]),
        ("Chula_RBC", 720, 576, ["RBC"]),
        ("Chula_RBC", 720, 576, []),
        ("Chula_RBC", 720, 576, ["WBC"]),
    ]

    for i, (dataset_name, w, h, classes) in enumerate(specs):
        img = UniversalImage(
            image_path=f"/data/{dataset_name}/img_{i}.jpg",
            dataset=dataset_name,
            width=w,
            height=h,
        )
        for cls in classes:
            img.objects.append(
                BoundingBox(class_id=0, class_name=cls, xc=0.5, yc=0.5, w=0.1, h=0.1)
            )
        ds.add(img)

    return ds


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_get_by_id_and_path():
    ds = _make_dataset()

    target = ds.images[2]

    assert ds.get_by_id(target.image_id) is target
    assert ds.get_by_path(target.image_path) is target
    assert ds.get_by_id("does-not-exist") is None


def test_contains_and_iteration():
    ds = _make_dataset()

    assert ds.images[0].image_id in ds
    assert "nonexistent" not in ds
    assert len(list(ds)) == len(ds)
    assert ds[0] is ds.images[0]


# ---------------------------------------------------------------------------
# Searching / Filtering
# ---------------------------------------------------------------------------

def test_find_predicate():
    ds = _make_dataset()

    # img_2 (800), img_3/img_4/img_5 (720 each) all satisfy >= 720
    wide = ds.find(lambda img: img.width >= 720)
    assert len(wide) == 4


def test_by_dataset_and_by_split():
    ds = _make_dataset()

    bccd = ds.by_dataset("BCCD")
    assert len(bccd) == 3
    assert all(img.dataset == "BCCD" for img in bccd)

    # original dataset is untouched by filtering
    assert len(ds) == 6


def test_by_class():
    ds = _make_dataset()

    wbc_images = ds.by_class("WBC")
    assert len(wbc_images) == 2

    platelet_images = ds.by_class("Platelet")
    assert len(platelet_images) == 1


def test_search_composed_query():
    ds = _make_dataset()

    # All three BCCD images (img_0: RBC+WBC, img_1: RBC, img_2: RBC+RBC+Platelet)
    # contain at least one RBC object, so all three should match.
    result = ds.search(dataset="BCCD", class_name="RBC")
    assert len(result) == 3


def test_search_max_objects():
    ds = _make_dataset()

    result = ds.search(max_objects=1)
    assert all(len(img.objects) <= 1 for img in result)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_defaults_and_updates():
    ds = _make_dataset()

    assert "created_at" in ds.metadata
    assert ds.get_metadata("version") == 1

    ds.set_metadata("project", "BloodCellAI")
    assert ds.get_metadata("project") == "BloodCellAI"
    assert ds.get_metadata("missing_key", "default") == "default"


def test_add_tag_deduplicates():
    ds = _make_dataset()

    ds.add_tag("multi-dataset")
    ds.add_tag("multi-dataset")

    assert ds.metadata["tags"].count("multi-dataset") == 1


# ---------------------------------------------------------------------------
# Statistics / Caching
# ---------------------------------------------------------------------------

def test_statistics_shape_and_values():
    ds = _make_dataset()

    stats = ds.statistics()

    assert stats["total_images"] == 6
    # img_0:2 + img_1:1 + img_2:3 + img_3:1 + img_4:0 + img_5:1 = 8
    assert stats["total_objects"] == 8
    assert stats["dataset_counts"] == {"BCCD": 3, "Chula_RBC": 3}
    assert stats["class_counts"]["RBC"] == 5


def test_statistics_cache_invalidated_on_add():
    ds = _make_dataset()

    stats1 = ds.statistics()
    stats2 = ds.statistics()
    assert stats1 is stats2  # same cached object, no recompute

    ds.add(UniversalImage(image_path="/data/BCCD/extra.jpg", dataset="BCCD", width=640, height=480))

    stats3 = ds.statistics()
    assert stats3 is not stats1
    assert stats3["total_images"] == 7


def test_class_balance_sums_to_one():
    ds = _make_dataset()

    balance = ds.class_balance()

    assert abs(sum(balance.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Train / Val / Test
# ---------------------------------------------------------------------------

def test_assign_splits_covers_every_image():
    ds = _make_dataset()

    ds.assign_splits(train=0.5, val=0.25, test=0.25, stratify_by=None, seed=0)

    train, val, test = ds.train_set(), ds.val_set(), ds.test_set()

    assert len(train) + len(val) + len(test) == len(ds)


def test_assign_splits_stratified_by_dataset():
    ds = _make_dataset()

    ds.assign_splits(train=0.34, val=0.33, test=0.33, stratify_by="dataset", seed=0)

    for name in ("BCCD", "Chula_RBC"):
        subset = ds.by_dataset(name)
        counts = subset.split_counts()
        assert sum(counts.values()) == len(subset)


def test_assign_splits_rejects_bad_proportions():
    ds = _make_dataset()

    try:
        ds.assign_splits(train=0.5, val=0.5, test=0.5)
        assert False, "expected ValueError"
    except ValueError:
        pass
