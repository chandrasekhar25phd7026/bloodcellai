
"""
==============================================================
BloodCellAI Enterprise Dataset Validator v2
==============================================================
"""

from PIL import Image


def validate_dataset(dataset):

    dataset.errors=[]

    dataset.warnings=[]

    dataset.statistics={}

    # ---------------------------------------------------
    # Corrupted Images
    # ---------------------------------------------------

    corrupted=0

    for image in dataset.image_files:

        try:

            Image.open(image).verify()

        except Exception:

            corrupted+=1

    dataset.statistics["CorruptedImages"]=corrupted

    dataset.statistics["Images"]=dataset.image_count

    dataset.statistics["Annotations"]=dataset.annotation_count

    annotation_type = dataset.annotation

    # ---------------------------------------------------
    # Detection Datasets
    # ---------------------------------------------------

    if dataset.task=="Detection":

        if annotation_type=="YOLO":

            if dataset.annotation_count==0:

                dataset.errors.append("YOLO labels missing")

            elif dataset.annotation_count<dataset.image_count:

                dataset.warnings.append(
                    "Some YOLO labels are missing"
                )

        elif annotation_type=="Pascal VOC":

            if dataset.annotation_count==0:

                dataset.errors.append("Pascal VOC XML annotations missing")

            elif dataset.annotation_count<dataset.image_count:

                dataset.warnings.append(
                    "Some Pascal VOC XML annotations are missing"
                )

        elif annotation_type=="COCO":

            # Like Malaria JSON, one (or a few, e.g. per-split) COCO
            # JSON file(s) describe the whole dataset -- a low
            # annotation_count here is expected and not itself a
            # problem, unlike per-image formats above.
            if dataset.annotation_count==0:

                dataset.errors.append(
                    "No COCO JSON file found for a COCO-annotated dataset"
                )

        elif annotation_type=="Chula TXT":

            dataset.warnings.append(
                "Sparse annotation dataset"
            )

        elif annotation_type=="Malaria JSON":

            # One or two JSON files describe
            # the whole dataset.
            pass

        elif annotation_type=="Generic TXT":

            dataset.warnings.append(
                "Generic TXT parser should verify annotation contents"
            )

        else:

            dataset.warnings.append(
                f"Unrecognized Detection annotation type "
                f"{annotation_type!r} -- no format-specific checks "
                "were run for this dataset."
            )

    # ---------------------------------------------------
    # Classification
    # ---------------------------------------------------

    elif dataset.task=="Classification":

        if annotation_type in [None,"None"]:

            dataset.warnings.append(
                "Labels managed externally"
            )

        elif annotation_type=="CSV":

            pass

    # ---------------------------------------------------
    # Clinical
    # ---------------------------------------------------

    elif dataset.task=="Clinical":

        if annotation_type=="CBC Report":

            pass

        else:

            dataset.warnings.append(
                "Clinical annotation requires review"
            )

    # ---------------------------------------------------
    # Unknown / unrecognized task (e.g. auto-detection
    # could not confidently classify this dataset)
    # ---------------------------------------------------

    else:

        dataset.warnings.append(
            f"Task {dataset.task!r} is not one of "
            "Detection/Classification/Clinical -- no format-specific "
            "checks were run. If this dataset was auto-registered, "
            "check DatasetInfo.registry['detection_evidence'] for why "
            "detection didn't confidently match a known format."
        )

    # ---------------------------------------------------
    # Final Status
    # ---------------------------------------------------

    if dataset.errors:

        dataset.status="ERROR"

    elif dataset.warnings:

        dataset.status="WARNING"

    else:

        dataset.status="READY"

    return dataset
