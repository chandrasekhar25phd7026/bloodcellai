
"""
==============================================================
BloodCellAI Enterprise File Matcher v2
==============================================================
"""

from pathlib import Path


class FileMatcher:

    def __init__(self):

        self.annotation_index = {}

        self.dataset_cache = set()

    def build_index(

        self,

        dataset_root,

        extensions=(".txt",".xml",".json",".csv")

    ):

        dataset_root = Path(dataset_root)

        if dataset_root in self.dataset_cache:

            return

        for ext in extensions:

            for f in dataset_root.rglob("*"+ext):

                self.annotation_index[f.stem] = f

        self.dataset_cache.add(dataset_root)

    def find_annotation(

        self,

        image_path,

    ):
        """
        Look up the annotation file matching `image_path` by filename
        stem, using the index already built by build_index().

        IMPORTANT: this used to call self.build_index() itself, using
        `dataset_root = image_path.parents[2]` as a guessed scope. For
        a typical layout (<project_root>/datasets/<DatasetName>/img.jpg),
        parents[2] is <project_root>/datasets -- the shared parent of
        EVERY dataset, not the specific one. That silently indexed
        annotation files from all datasets into one flat dict keyed
        only by filename stem, so two datasets sharing a stem (e.g.
        both having a "sample1.txt") would silently cross-contaminate:
        an image from dataset A could get matched to dataset B's
        annotation file, with no warning or error at all. Confirmed by
        building "RBCMorphology" and "Malaria" test data with matching
        filenames -- RBCMorphology's image was silently matched to
        Malaria's annotation file.

        Fixed to just trust the index the caller already built via an
        explicit build_index(<the correct dataset folder>) call --
        which universal_builder.py already does, once per dataset,
        with the correct scope -- instead of re-guessing it here.
        """

        image_path = Path(image_path)

        return self.annotation_index.get(image_path.stem, None)
