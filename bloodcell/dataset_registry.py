
"""
==========================================================
BloodCellAI Enterprise Dataset Registry v2
==========================================================
"""

DATASET_REGISTRY = {

    "BCCD": {

        "id": "DS004",

        "task": "Detection",

        "annotation": "Pascal VOC",

        "image_extension": ".jpg",

        "label_extension": ".xml",

        "classes": {

            0: "RBC",
            1: "WBC",
            2: "Platelet"

        },

        "source": "Kaggle",

        "year": 2020,

        "license": "Public"

    },

    "Chula_RBC": {

        "id": "DS005",

        "task": "Detection",

        "annotation": "Chula TXT",

        "image_extension": ".jpg",

        "label_extension": ".txt",

        "classes": {

            0: "Normal_RBC",
            1: "Macrocyte",
            2: "Microcyte",
            3: "Spherocyte",
            4: "Target_Cell",
            5: "Stomatocyte",
            6: "Ovalocyte",
            7: "Tear_Drop_Cell",
            8: "Burr_Cell",
            9: "Schistocyte",
            10: "Uncategorized",
            11: "Hypochromia",
            12: "Elliptocyte"

        },

        "source": "Chulalongkorn University",

        "year": 2021,

        "license": "Academic"

    },

    "Malaria Bounding Boxes": {

        "id": "DS008",

        "task": "Detection",

        "annotation": "Malaria JSON",

        "image_extension": ".png",

        "label_extension": ".json",

        "classes": {

            0: "RBC",
            1: "WBC",
            3: "Parasite"

        },

        "source": "NIH"

    },

    "Malaria": {

        "id": "DS007",

        "task": "Detection",

        "annotation": "Generic TXT",

        "image_extension": ".jpg",

        "label_extension": ".txt",

        "classes": {

            0: "RBC",
            1: "WBC",
            3: "Parasite"

        }

    },

    "NIH_Malaria": {

        "id": "DS009",

        "task": "Classification",

        "annotation": "None",

        "image_extension": ".png",

        "classes": {

            0: "Parasitized",
            1: "Uninfected"

        }

    },

    "AcuteLeukemia": {

        "id": "DS001",

        "task": "Classification",

        "annotation": "CSV",

        "image_extension": ".bmp",

        "classes": {

            0: "Blast"

        }

    },

    "ALL_IDB": {

        "id": "DS002",

        "task": "Classification",

        "annotation": "None",

        "image_extension": ".jpg",

        "classes": {

            0: "Blast"

        }

    },

    "ALL_IDB2": {

        "id": "DS003",

        "task": "Classification",

        "annotation": "None",

        "image_extension": ".tif",

        "classes": {

            0: "Blast"

        }

    },

    "LISC": {

        "id": "DS006",

        "task": "Classification",

        "annotation": "None",

        "image_extension": ".bmp",

        "classes": {

            0: "Neutrophil",
            1: "Lymphocyte",
            2: "Monocyte",
            3: "Eosinophil",
            4: "Basophil"

        }

    },

    "Raabin": {

        "id": "DS011",

        "task": "Classification",

        "annotation": "None",

        "image_extension": ".jpg",

        "classes": {

            0: "Neutrophil",
            1: "Lymphocyte",
            2: "Monocyte",
            3: "Eosinophil",
            4: "Basophil"

        }

    },

    "Raabin_WBC": {

        "id": "DS012",

        "task": "Classification",

        "annotation": "None",

        "image_extension": ".jpg",

        "classes": {

            0: "Neutrophil",
            1: "Lymphocyte",
            2: "Monocyte",
            3: "Eosinophil",
            4: "Basophil"

        }

    },

    "Platelets": {

        "id": "DS010",

        "task": "Classification",

        "annotation": "None",

        "image_extension": ".jpg",

        "classes": {

            0: "Platelet"

        }

    },

    "RBCMorphology": {

        "id": "DS013",

        "task": "Clinical",

        "annotation": "CBC Report",

        "image_extension": ".png",

        "label_extension": ".txt",

        "classes": {

            0: "Normal",
            1: "Microcytic",
            2: "Hypochromic",
            3: "Elliptocyte",
            4: "Polychromasia"

        }

    }

}


# ============================================================
# Registry API
# ============================================================

def get_classes(dataset_name):
    """
    Return class dictionary for a dataset.

    Parameters
    ----------
    dataset_name : str

    Returns
    -------
    dict
    """

    info = DATASET_REGISTRY.get(dataset_name)

    if info is None:
        return {}

    return info.get("classes", {})


def get_annotation_type(dataset_name):
    """
    Return the "annotation" field for a dataset (e.g. "Pascal VOC",
    "Chula TXT", "Malaria JSON", "CSV", or "None" for folder-per-class
    classification datasets that have no per-image annotation file at
    all -- the class comes from the image's parent folder name
    instead).

    Returns
    -------
    str or None
    """

    info = DATASET_REGISTRY.get(dataset_name)

    if info is None:
        return None

    return info.get("annotation")