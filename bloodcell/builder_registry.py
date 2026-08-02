
"""
==============================================================
BloodCellAI Enterprise Builder Registry v2
==============================================================
"""

from .adapters import *

# ------------------------------------------------------------
# Enterprise Dataset Registry
# ------------------------------------------------------------

BUILDER_REGISTRY = {

    "BCCD":{

        "adapter":PascalVOCAdapter(),

        "task":"Detection",

        "annotation":"Pascal VOC"

    },

    "Chula_RBC":{

        "adapter":ChulaAdapter(),

        "task":"Detection",

        "annotation":"Chula TXT"

    },

    "Malaria":{

        "adapter":YOLOAdapter(),

        "task":"Detection",

        "annotation":"Generic TXT"

    },

    "Malaria Bounding Boxes":{

        "adapter":MalariaAdapter(),

        "task":"Detection",

        "annotation":"Malaria JSON"

    },

    "RBCMorphology":{

        "adapter":RBCMorphologyAdapter(),

        "task":"Clinical",

        "annotation":"CBC Report"

    },

    # ------------------------------------------------------------
    # Classification-task datasets.
    #
    # These build via UniversalBuilder.build_classification_dataset(),
    # NOT the detection-oriented build_dataset()/pipeline.py path --
    # there is no per-image annotation file for FileMatcher to find.
    # The "adapter" entry here is for introspection consistency
    # (get_adapter() etc.) and documents which adapter each dataset's
    # actual build path constructs.
    # ------------------------------------------------------------

    "LISC":{

        "adapter":ClassificationAdapter(),

        "task":"Classification",

        "annotation":"None"

    },

    "NIH_Malaria":{

        "adapter":ClassificationAdapter(),

        "task":"Classification",

        "annotation":"None"

    },

    "ALL_IDB":{

        "adapter":ClassificationAdapter(),

        "task":"Classification",

        "annotation":"None"

    },

    "ALL_IDB2":{

        "adapter":ClassificationAdapter(),

        "task":"Classification",

        "annotation":"None"

    },

    "Raabin":{

        "adapter":ClassificationAdapter(),

        "task":"Classification",

        "annotation":"None"

    },

    "Raabin_WBC":{

        "adapter":ClassificationAdapter(),

        "task":"Classification",

        "annotation":"None"

    },

    "Platelets":{

        "adapter":ClassificationAdapter(),

        "task":"Classification",

        "annotation":"None"

    },

    "AcuteLeukemia":{

        "adapter":ClassificationAdapter(),

        "task":"Classification",

        "annotation":"CSV"

    }

}


def get_builder(dataset):

    return BUILDER_REGISTRY.get(dataset,None)


def get_adapter(dataset):

    builder=get_builder(dataset)

    if builder is None:

        return None

    return builder["adapter"]


def get_task(dataset):

    builder=get_builder(dataset)

    if builder is None:

        return None

    return builder["task"]


def get_annotation(dataset):

    builder=get_builder(dataset)

    if builder is None:

        return None

    return builder["annotation"]
