
"""
==============================================================
BloodCellAI Enterprise Processing Pipeline
==============================================================
"""

import time

from .builder_registry import get_adapter


def _run_preprocessing(image_path, ui, preprocessing_manager, logger, dataset_name):
    """
    Run the configured preprocessing pipeline against one image and
    attach the results to its UniversalImage.metadata under the
    "preprocessing" key.

    A preprocessing failure (e.g. a corrupt/unreadable image file) is
    recorded in metadata, not raised -- the annotation itself already
    built successfully, so the image stays in the dataset with a
    flagged quality issue rather than being silently dropped. This is
    what actually makes "automatically validated and preprocessed
    during dataset creation" true: every image gets a quality record,
    whether it passed or not.
    """

    try:

        result = preprocessing_manager.preprocess_image(str(image_path))

        ui.metadata["preprocessing"] = {
            "passed": result.passed,
            "quality_score": result.quality_metrics.quality_score,
            "quality_metrics": result.quality_metrics.to_dict(),
            "warnings": list(result.warnings),
            "transforms_applied": [
                record.name for record in result.transform_history
                if record.successful
            ],
        }

    except Exception as exc:

        ui.metadata["preprocessing"] = {
            "passed": False,
            "error": str(exc),
        }


def build_single_image(
    image_path,
    dataset_name,
    dataset,
    logger,
    matcher,
    adapter=None,
    annotation_file=None,
    preprocessing_manager=None,
    **kwargs
):
    """
    Build one UniversalImage from a dataset image.

    Parameters
    ----------
    image_path : Path
        Image file

    dataset_name : str
        Dataset name

    dataset : UniversalDataset
        Global dataset object

    logger : BuildLogger
        Build logger

    matcher : FileMatcher
        Annotation matcher

    adapter : BaseAdapter, optional
        Explicit adapter to use. If not given (existing behavior),
        resolved by dataset_name via get_adapter()/BUILDER_REGISTRY.
        Passing one explicitly is what lets auto-detected datasets --
        whose name was never pre-registered in BUILDER_REGISTRY -- be
        built at all, by resolving an adapter from the *detected
        annotation type* instead (see
        universal_builder.UniversalBuilder.build_from_info()).

    annotation_file : Path, optional
        Explicit annotation file to use for this image, bypassing
        matcher.find_annotation()'s per-image filename-stem lookup.
        Needed for "whole-dataset" annotation formats (COCO, Malaria
        JSON) where ONE file describes every image in the dataset --
        stem-based matching can only ever find such a file by
        coincidence (image and annotation happening to share a stem),
        not by design. See
        universal_builder.UniversalBuilder.build_from_info(), which
        locates the single shared file once and passes it here for
        every image.

    preprocessing_manager : preprocessing.PreprocessingManager, optional
        If given, every successfully-built image is run through this
        manager's preprocessing/quality pipeline, and the result
        (quality metrics, pass/fail, applied transforms) is attached
        to the image's `.metadata["preprocessing"]`. This is what
        wires preprocessing into dataset loading -- see
        universal_builder.UniversalBuilder.enable_preprocessing().

    Returns
    -------
    UniversalImage or None
    """

    start = time.time()

    try:

        # ------------------------------------------------------
        # Get Adapter
        # ------------------------------------------------------

        if adapter is None:
            adapter = get_adapter(dataset_name)

        if adapter is None:

            logger.failure(
                dataset=dataset_name,
                image=image_path.name,
                message=f"No adapter registered for '{dataset_name}'",
                elapsed=time.time() - start
            )

            return None

        # ------------------------------------------------------
        # Find Annotation
        # ------------------------------------------------------

        annotation = annotation_file

        if annotation is None:
            annotation = matcher.find_annotation(image_path)

        if annotation is None:

            logger.failure(
                dataset=dataset_name,
                image=image_path.name,
                message="Annotation missing (image skipped)",
                elapsed=time.time() - start
            )

            return None

        # ------------------------------------------------------
        # Convert to UniversalImage
        # ------------------------------------------------------

        ui = adapter.convert(
            annotation_file=annotation,
            image_path=str(image_path),
            dataset=dataset_name,
            **kwargs
        )

        # ------------------------------------------------------
        # Preprocess / Validate (optional)
        # ------------------------------------------------------

        if preprocessing_manager is not None:

            _run_preprocessing(
                image_path, ui, preprocessing_manager, logger, dataset_name
            )

        # ------------------------------------------------------
        # Add to Dataset
        # ------------------------------------------------------

        dataset.add(ui)

        # ------------------------------------------------------
        # Log Success
        # ------------------------------------------------------

        logger.success(
            dataset=dataset_name,
            image=image_path.name,
            objects=len(getattr(ui, "objects", []) or []),
            elapsed=time.time() - start
        )

        return ui

    except Exception as e:

        logger.failure(
            dataset=dataset_name,
            image=image_path.name,
            message=str(e),
            elapsed=time.time() - start
        )

        return None