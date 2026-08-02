from pathlib import Path
import time
import json

from .dataset_registry import DATASET_REGISTRY
from .file_matcher import FileMatcher
from .pipeline import build_single_image, _run_preprocessing
from .universal_dataset import UniversalDataset
from .build_logger import BuildLogger
from .adapters import (
    ClassificationAdapter,
    YOLOAdapter,
    PascalVOCAdapter,
    CocoAdapter,
    ChulaAdapter,
    MalariaAdapter,
)


# Maps a *detected annotation type string* (as produced by
# annotation_intelligence.detect_dataset_format(), and used
# throughout dataset_registry.py's "annotation" field) to the adapter
# class that handles it. This is what lets build_from_info() resolve
# an adapter generically for an auto-detected dataset, instead of
# requiring the dataset's name to already be a hand-curated entry in
# builder_registry.BUILDER_REGISTRY.
ANNOTATION_TYPE_ADAPTERS = {
    "YOLO": YOLOAdapter,
    "Pascal VOC": PascalVOCAdapter,
    "COCO": CocoAdapter,
    "Chula TXT": ChulaAdapter,
    "Malaria JSON": MalariaAdapter,
}

# Annotation formats where ONE file describes every image in the
# dataset, rather than one file per image. FileMatcher's per-image
# filename-stem matching cannot find this kind of file by design (only
# by a filename coincidence) -- these need the shared file located
# once and passed explicitly to every image instead. Confirmed by
# actually running a real COCO-shaped build: it silently built 0
# images until this was fixed.
WHOLE_DATASET_ANNOTATION_FORMATS = {"COCO", "Malaria JSON"}


def _find_whole_dataset_annotation_file(dataset_folder, annotation_type):
    """
    Locate the annotation file(s) describing every image in a
    whole-dataset-annotation-format dataset (COCO, Malaria JSON), and
    return a single path to use.

    Some real datasets split their whole-dataset annotations across
    multiple files rather than one (e.g. the real Malaria Bounding
    Boxes dataset ships training.json + test.json, each covering a
    different, non-overlapping set of images) -- taking only the
    first matching file, as an earlier version of this function did,
    silently discards every record in the others. Confirmed on real
    data: doing that dropped 1,208 of the dataset's real annotated
    records (80,113 real objects), leaving the vast majority of
    images with zero objects after harmonization, with no error or
    warning at all.

    When more than one matching file is found, their records are
    merged into a single combined file (written once, cached for
    reuse) so every image's real annotations are available regardless
    of which of the original files it came from.

    Returns
    -------
    Path or None
    """

    from .annotation_intelligence import is_coco_json, is_malaria_json

    json_files = sorted(Path(dataset_folder).rglob("*.json"))

    # Exclude any previously-created merged file from re-matching itself
    # on a second call.
    json_files = [f for f in json_files if f.name != "_merged_whole_dataset.json"]

    if annotation_type == "COCO":
        matches = [f for f in json_files if is_coco_json(f)]
    elif annotation_type == "Malaria JSON":
        matches = [f for f in json_files if is_malaria_json(f)]
    else:
        matches = []

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    merged_path = Path(dataset_folder) / "_merged_whole_dataset.json"

    if merged_path.exists():
        return merged_path

    if annotation_type == "Malaria JSON":
        # Each file is a JSON list of per-image records -- concatenate them.
        merged_records = []
        for f in matches:
            with open(f, "r") as fh:
                data = json.load(fh)
            merged_records.extend(data if isinstance(data, list) else [data])

        with open(merged_path, "w") as fh:
            json.dump(merged_records, fh)

    elif annotation_type == "COCO":
        # Each file is a COCO dict with "images"/"annotations"/"categories" --
        # concatenate images/annotations, keep categories from the first
        # file (COCO category ids are expected to be consistent across a
        # dataset's own splits).
        merged = {"images": [], "annotations": [], "categories": []}
        for f in matches:
            with open(f, "r") as fh:
                data = json.load(fh)
            merged["images"].extend(data.get("images", []))
            merged["annotations"].extend(data.get("annotations", []))
            if not merged["categories"]:
                merged["categories"] = data.get("categories", [])

        with open(merged_path, "w") as fh:
            json.dump(merged, fh)

    return merged_path


class UniversalBuilder:

    def __init__(self, project_root):

        self.project_root = Path(project_root)

        self.dataset = UniversalDataset()

        self.logger = BuildLogger()

        self.statistics = {}

        self.preprocessing_manager = None

    def enable_preprocessing(self, config=None):
        """
        Turn on automatic preprocessing/quality-assessment during
        dataset creation: every image built after calling this goes
        through the preprocessing pipeline, with the results (quality
        score, pass/fail, applied transforms) attached to
        UniversalImage.metadata["preprocessing"].

        Parameters
        ----------
        config : preprocessing.PreprocessingConfig, optional
            Defaults to a fresh PreprocessingConfig() (its own
            defaults) if not given.
        """

        from preprocessing.preprocessing_config import PreprocessingConfig
        from preprocessing.preprocessing_manager import PreprocessingManager

        if config is None:
            config = PreprocessingConfig()

        self.preprocessing_manager = PreprocessingManager(config)

        return self.preprocessing_manager

    def disable_preprocessing(self):
        """
        Turn automatic preprocessing back off.
        """

        self.preprocessing_manager = None

    def build_from_info(self, dataset_info):
        """
        Build (harmonize into self.dataset) a dataset described by a
        DatasetInfo object -- the object produced by
        dataset_builder.UniversalDatasetBuilder.register_dataset() /
        auto_register(), typically after also calling .prepare() on
        it (scan + pre-build validation).

        Unlike build_dataset(name), this does NOT require the
        dataset's name to be a pre-existing entry in
        builder_registry.BUILDER_REGISTRY -- the adapter is resolved
        from dataset_info.annotation (the detected/registered
        annotation type) via ANNOTATION_TYPE_ADAPTERS instead. This
        is what actually completes the "point BloodCellAI at an
        arbitrary folder" story: detect_dataset_format() ->
        auto_register() -> prepare() -> build_from_info() all work
        together without the dataset ever needing a hand-written
        DATASET_REGISTRY/BUILDER_REGISTRY entry.

        Classification-task datasets (annotation "None" or "CSV")
        are delegated to the same logic build_classification_dataset()
        already uses.

        Parameters
        ----------
        dataset_info : bloodcell.dataset_info.DatasetInfo

        Returns
        -------
        tuple
            (self.dataset, built) -- same shape as build_dataset().
        """

        if dataset_info.task == "Classification":
            return self._build_classification_from_info(dataset_info)

        adapter_cls = ANNOTATION_TYPE_ADAPTERS.get(dataset_info.annotation)

        if adapter_cls is None:
            raise ValueError(
                f"No adapter available for annotation type "
                f"{dataset_info.annotation!r} (dataset "
                f"{dataset_info.name!r}). Known types: "
                f"{sorted(ANNOTATION_TYPE_ADAPTERS)}. If this dataset "
                "was auto-detected as 'Unknown', it needs to be "
                "registered manually instead."
            )

        # Adapters resolve class names via dataset_registry.get_classes(),
        # which only knows names already in the global DATASET_REGISTRY.
        # For an auto-detected dataset (not a hand-curated registry
        # entry), inject its detected classes under its own name so
        # existing adapters resolve real class names instead of
        # falling back to generic "Class_0"/"Class_1" placeholders.
        if dataset_info.name not in DATASET_REGISTRY:
            DATASET_REGISTRY[dataset_info.name] = {
                "task": dataset_info.task,
                "annotation": dataset_info.annotation,
                "classes": dataset_info.registry.get("classes", {}),
            }

        adapter = adapter_cls()

        dataset_folder = dataset_info.path

        matcher = FileMatcher()
        matcher.build_index(dataset_folder)

        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            images.extend(dataset_folder.rglob(ext))
        images = sorted(images)

        # Whole-dataset annotation formats (COCO, Malaria JSON) describe
        # every image in ONE shared file -- FileMatcher's per-image
        # filename-stem matching can only find such a file by
        # coincidence (image and annotation happening to share a
        # stem), not by design, so it's located once here and passed
        # explicitly for every image instead.
        shared_annotation_file = None

        if dataset_info.annotation in WHOLE_DATASET_ANNOTATION_FORMATS:

            shared_annotation_file = _find_whole_dataset_annotation_file(
                dataset_folder, dataset_info.annotation
            )

            if shared_annotation_file is None:
                raise ValueError(
                    f"Dataset {dataset_info.name!r} is annotated as "
                    f"{dataset_info.annotation!r}, which needs one "
                    "shared annotation file describing every image, "
                    f"but none was found under {dataset_folder}."
                )

        start = time.time()
        before = len(self.dataset)

        for image in images:

            build_single_image(
                image_path=image,
                dataset_name=dataset_info.name,
                matcher=matcher,
                dataset=self.dataset,
                logger=self.logger,
                adapter=adapter,
                annotation_file=shared_annotation_file,
                preprocessing_manager=self.preprocessing_manager,
            )

        elapsed = time.time() - start
        built = len(self.dataset) - before

        self.statistics[dataset_info.name] = {
            "ImagesFound": len(images),
            "ImagesBuilt": built,
            "Elapsed": elapsed,
        }

        return self.dataset, built

    def _build_classification_from_info(self, dataset_info):
        """
        Classification counterpart to build_from_info() -- delegates
        to build_classification_dataset()'s existing folder-per-class
        / CSV logic, after making sure the auto-detected classes are
        available under this dataset's name the same way
        build_from_info() does for detection datasets.
        """

        if dataset_info.name not in DATASET_REGISTRY:
            DATASET_REGISTRY[dataset_info.name] = {
                "task": dataset_info.task,
                "annotation": dataset_info.annotation,
                "classes": dataset_info.registry.get("classes", {}),
            }

        return self.build_classification_dataset(
            dataset_info.name,
            dataset_folder=dataset_info.path,
        )

    def build_dataset(self, dataset_name):

        dataset_folder = self.project_root / "datasets" / dataset_name

        matcher = FileMatcher()

        matcher.build_index(dataset_folder)

        images = []

        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            images.extend(dataset_folder.rglob(ext))

        images = sorted(images)

        print(f"\nBuilding {dataset_name}")
        print(f"Images : {len(images)}")

        start = time.time()

        before = len(self.dataset)

        for image in images:

            build_single_image(
                image_path=image,
                dataset_name=dataset_name,
                matcher=matcher,
                dataset=self.dataset,
                logger=self.logger,
                preprocessing_manager=self.preprocessing_manager,
            )

        elapsed = time.time() - start

        built = len(self.dataset) - before

        self.statistics[dataset_name] = {
            "ImagesFound": len(images),
            "ImagesBuilt": built,
            "Elapsed": elapsed
        }

        return self.dataset, built

    def build_classification_dataset(self, dataset_name, dataset_folder=None):
        """
        Build a classification-task dataset.

        Unlike detection datasets, classification datasets have no
        per-image annotation file for FileMatcher to find -- the label
        is either the image's parent folder name (registry
        annotation == "None", e.g. LISC: <root>/Neutrophil/img1.bmp)
        or looked up from a single dataset-wide CSV file (registry
        annotation == "CSV", e.g. AcuteLeukemia:
        <root>/images/1.bmp + <root>/labels.csv). This needs its own
        build path rather than reusing build_dataset()/pipeline.py,
        which assume one annotation file per image.

        Parameters
        ----------
        dataset_name : str

        dataset_folder : str or Path, optional
            Explicit path to the dataset's folder. Defaults to the
            standard `<project_root>/datasets/<dataset_name>`
            convention (existing behavior) when not given -- pass
            this explicitly for a dataset registered from an
            arbitrary path (see
            dataset_builder.UniversalDatasetBuilder.auto_register()
            and build_from_info(), which do exactly that).
        """

        info = DATASET_REGISTRY.get(dataset_name)

        if info is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        annotation_style = info.get("annotation", "None")

        classes = info.get("classes", {})

        if dataset_folder is None:
            dataset_folder = self.project_root / "datasets" / dataset_name
        else:
            dataset_folder = Path(dataset_folder)

        adapter = ClassificationAdapter()

        before = len(self.dataset)

        start = time.time()

        if annotation_style == "CSV":
            self._build_csv_classification(
                dataset_folder, dataset_name, classes, adapter
            )
        else:
            self._build_folder_classification(
                dataset_folder, dataset_name, classes, adapter
            )

        elapsed = time.time() - start

        built = len(self.dataset) - before

        self.statistics[dataset_name] = {
            "ImagesBuilt": built,
            "Elapsed": elapsed
        }

        return self.dataset, built

    @staticmethod
    def _normalize_class_name(name):
        """
        Normalize a class name for matching folder names / CSV values
        against the registry, tolerating case and separator
        differences (e.g. "neutrophil", "Neutrophil", "NEUTROPHIL").
        """

        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    def _build_folder_classification(self, dataset_folder, dataset_name, classes, adapter):
        """
        Build from a folder-per-class layout:
        <dataset_folder>/<ClassName>/image.ext
        """

        from PIL import Image as PILImage

        name_to_id = {
            self._normalize_class_name(name): cid
            for cid, name in classes.items()
        }

        if not dataset_folder.is_dir():
            return

        image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        for class_folder in sorted(dataset_folder.iterdir()):

            if not class_folder.is_dir():
                continue

            normalized = self._normalize_class_name(class_folder.name)

            class_id = name_to_id.get(normalized)

            if class_id is None:

                self.logger.failure(
                    dataset=dataset_name,
                    image=class_folder.name,
                    message=(
                        f"Folder name {class_folder.name!r} does not "
                        f"match any registered class for {dataset_name}"
                    ),
                    elapsed=0.0,
                )

                continue

            for image_path in sorted(class_folder.iterdir()):

                if image_path.suffix.lower() not in image_exts:
                    continue

                start = time.time()

                try:

                    with PILImage.open(image_path) as im:
                        width, height = im.size

                    ui = adapter.convert(
                        image_path=str(image_path),
                        class_id=class_id,
                        dataset=dataset_name,
                        width=width,
                        height=height,
                    )

                    if self.preprocessing_manager is not None:
                        _run_preprocessing(
                            image_path, ui, self.preprocessing_manager,
                            self.logger, dataset_name,
                        )

                    self.dataset.add(ui)

                    self.logger.success(
                        dataset=dataset_name,
                        image=image_path.name,
                        objects=len(ui.objects),
                        elapsed=time.time() - start,
                    )

                except Exception as e:

                    self.logger.failure(
                        dataset=dataset_name,
                        image=image_path.name,
                        message=str(e),
                        elapsed=time.time() - start,
                    )

    def _build_csv_classification(self, dataset_folder, dataset_name, classes, adapter):
        """
        Build from a single dataset-wide CSV mapping image id -> class
        label, e.g. AcuteLeukemia: one CSV covering every image, not
        one annotation file per image.

        Tolerates common column-name variants ("image ID"/"image_id"/
        "filename", "class label"/"label"/"class") rather than
        requiring one exact header.
        """

        from PIL import Image as PILImage
        import csv

        csv_files = list(dataset_folder.rglob("*.csv"))

        if not csv_files:

            self.logger.failure(
                dataset=dataset_name,
                image="",
                message=f"No CSV file found under {dataset_folder}",
                elapsed=0.0,
            )

            return

        image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        image_index = {}
        for ext in image_exts:
            for f in dataset_folder.rglob("*" + ext):
                image_index[f.stem] = f
                # Also index a leading-zero-stripped form for numeric
                # stems, since CSV image-id columns and actual
                # filenames often disagree on zero-padding (e.g. CSV
                # id "1" vs filename "001.bmp") -- confirmed with a
                # real public CSV-labeled WBC dataset during testing.
                if f.stem.isdigit():
                    image_index.setdefault(str(int(f.stem)), f)

        def _find_column(fieldnames, candidates):
            for field in fieldnames:
                normalized = field.strip().lower()
                if any(c in normalized for c in candidates):
                    return field
            return None

        for csv_file in csv_files:

            with open(csv_file, "r", newline="") as f:

                reader = csv.DictReader(f)

                id_col = _find_column(reader.fieldnames or [], ("image", "filename", "id"))
                label_col = _find_column(reader.fieldnames or [], ("class", "label"))

                if id_col is None or label_col is None:

                    self.logger.failure(
                        dataset=dataset_name,
                        image=csv_file.name,
                        message=(
                            f"Could not identify image-id/class-label "
                            f"columns in {csv_file.name} "
                            f"(found: {reader.fieldnames})"
                        ),
                        elapsed=0.0,
                    )

                    continue

                for row in reader:

                    start = time.time()

                    image_stem = str(row[id_col]).strip()

                    image_path = image_index.get(image_stem)

                    if image_path is None and image_stem.isdigit():
                        image_path = image_index.get(str(int(image_stem)))

                    if image_path is None:

                        self.logger.failure(
                            dataset=dataset_name,
                            image=image_stem,
                            message="Annotation missing (image skipped)",
                            elapsed=0.0,
                        )

                        continue

                    try:

                        class_id = int(row[label_col])

                        with PILImage.open(image_path) as im:
                            width, height = im.size

                        ui = adapter.convert(
                            image_path=str(image_path),
                            class_id=class_id,
                            dataset=dataset_name,
                            width=width,
                            height=height,
                        )

                        if self.preprocessing_manager is not None:
                            _run_preprocessing(
                                image_path, ui, self.preprocessing_manager,
                                self.logger, dataset_name,
                            )

                        self.dataset.add(ui)

                        self.logger.success(
                            dataset=dataset_name,
                            image=image_path.name,
                            objects=len(ui.objects),
                            elapsed=time.time() - start,
                        )

                    except Exception as e:

                        self.logger.failure(
                            dataset=dataset_name,
                            image=image_path.name,
                            message=str(e),
                            elapsed=time.time() - start,
                        )

    def build_many(self, dataset_names):

        total_built = 0

        for ds in dataset_names:

            _, built = self.build_dataset(ds)

            total_built += built

        return self.dataset, total_built

    def build_all(self):

        total_built = 0

        for ds in DATASET_REGISTRY.keys():

            try:

                _, built = self.build_dataset(ds)

                total_built += built

            except Exception as e:

                print(f"{ds} FAILED : {e}")

        return self.dataset, total_built

    def summary(self):

        return self.dataset.summary()

    def report(self):

        return self.statistics

    def log(self):

        return self.logger.dataframe()