"""
BloodCellAI
stain_normalization.py

Section 1 (Foundation)
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .preprocessing_models import TransformRecord, TransformStatus
from .preprocessing_config import PreprocessingConfig
from transforms.base_transform import BaseTransform
from transforms.registry import TransformRegistry


class StainNormalizationMethod(Enum):
    NONE = "none"
    REINHARD = "reinhard"
    MACENKO = "macenko"
    HISTOGRAM = "histogram"


class StainNormalizationTransform(BaseTransform):

    def __init__(self, config: PreprocessingConfig) -> None:
        super().__init__()
        self._config = config
        self._logger = logging.getLogger(self.__class__.__name__)

        stain = getattr(config, "stain_normalization", None)
        if stain is None:
            raise ValueError("Stain normalization configuration is missing.")

        self._enabled = stain.enabled
        self._method = StainNormalizationMethod(stain.method)
        self._reference_image = getattr(stain, "reference_image", None)
        self._preserve_luminance = getattr(stain, "preserve_luminance", True)
        self._standardize_brightness_enabled = getattr(stain, "standardize_brightness", True)
        self._alpha = getattr(stain, "alpha", 1.0)
        self._beta = getattr(stain, "beta", 0.15)
        self._background_threshold = getattr(stain, "background_threshold", 240)

        self._last_stain_matrix = None
        self._last_concentration_matrix = None
        self._last_statistics = None

    @property
    def config(self) -> PreprocessingConfig:
        """
        Expose the preprocessing config under the `config` name used
        throughout this class (`self.config.stain_normalization...`)
        -- only `self._config` was ever set in __init__, so every one
        of those existing references would raise AttributeError the
        moment they actually ran.
        """
        return self._config

    def validate_input(self, image: np.ndarray) -> None:
        if image is None:
            raise ValueError("Image is None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected numpy.ndarray.")
        if image.size == 0:
            raise ValueError("Image is empty.")
        if image.ndim != 3:
            raise ValueError("Expected BGR colour image.")
        if image.dtype != np.uint8:
            raise TypeError("Expected uint8 image.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "method": self._method.value,
            "preserve_luminance": self._preserve_luminance,
            "standardize_brightness": self._standardize_brightness_enabled,
            "alpha": self._alpha,
            "beta": self._beta,
            "background_threshold": self._background_threshold,
        }

    @property
    def method(self):
        return self._method

    @property
    def stain_matrix(self):
        return self._last_stain_matrix

    @property
    def concentration_matrix(self):
        return self._last_concentration_matrix

    @property
    def statistics(self):
        return self._last_statistics

    def reset(self) -> None:
        self._last_stain_matrix = None
        self._last_concentration_matrix = None
        self._last_statistics = None

# -------------------------------------------------------------------------
# SECTION 2
# Continue directly after reset()
# -------------------------------------------------------------------------

    def _ensure_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)

    def _bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _rgb_to_bgr(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _bgr_to_lab(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def _lab_to_bgr(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image[image == 0] = 1
        return -np.log(image / 255.0)

    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        rgb = 255.0 * np.exp(-od)
        rgb = np.clip(rgb, 0, 255)
        return rgb.astype(np.uint8)

    def _remove_background(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray < self._background_threshold
        pixels = image[mask]
        return pixels, mask

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return matrix / norm

    def _calculate_channel_statistics(self, image: np.ndarray) -> dict:
        stats = {}
        for index, name in enumerate(("B", "G", "R")):
            channel = image[:, :, index]
            stats[name] = {
                "mean": float(np.mean(channel)),
                "std": float(np.std(channel)),
                "min": int(np.min(channel)),
                "max": int(np.max(channel)),
            }
        self._last_statistics = stats
        return stats

    def _standardize_brightness(self, image: np.ndarray) -> np.ndarray:
        if not self._standardize_brightness_enabled:
            return image
        lab = self._bgr_to_lab(image)
        l, a, b = cv2.split(lab)
        percentile = np.percentile(l, 90)
        if percentile > 0:
            scale = 255.0 / percentile
            l = np.clip(l * scale, 0, 255).astype(np.uint8)
        lab = cv2.merge((l, a, b))
        return self._lab_to_bgr(lab)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        self.validate_input(image)
        image = self._ensure_uint8(image)
        image = self._standardize_brightness(image)
        return image

# ---------------------- End of Section 2 ----------------------
    # =====================================================================
    # Reference Image Utilities
    # =====================================================================

    def _validate_reference_image(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the stain reference image.
        """

        if image is None:
            raise ValueError(
                "Reference image cannot be None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Reference image must be a numpy.ndarray."
            )

        if image.ndim != 3:
            raise ValueError(
                "Reference image must be a color image."
            )

        if image.shape[2] != 3:
            raise ValueError(
                "Reference image must contain three channels."
            )

        if image.dtype != np.uint8:
            raise TypeError(
                "Reference image must be uint8."
            )

        if image.size == 0:
            raise ValueError(
                "Reference image is empty."
            )

    # ---------------------------------------------------------------------

    def _load_reference_image(
        self,
    ) -> np.ndarray:
        """
        Load the reference stain image.

        Returns
        -------
        numpy.ndarray
            Reference BGR image.
        """

        if self._reference_image is None:

            raise ValueError(
                "No reference image specified for "
                "stain normalization."
            )

        #
        # Already loaded image
        #

        if isinstance(
            self._reference_image,
            np.ndarray,
        ):

            reference = self._reference_image.copy()

            self._validate_reference_image(
                reference,
            )

            return reference

        #
        # Image path
        #

        image_path = Path(
            self._reference_image,
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Reference image not found: "
                f"{image_path}"
            )

        reference = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if reference is None:

            raise ValueError(
                "Unable to load reference image."
            )

        self._validate_reference_image(
            reference,
        )

        return reference

    # ---------------------------------------------------------------------

    def _compute_mean_std(
        self,
        channel: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute channel mean and standard deviation.
        """

        channel = channel.astype(
            np.float32,
        )

        mean = float(
            np.mean(channel)
        )

        std = float(
            np.std(channel)
        )

        #
        # Prevent division by zero
        #

        if std < 1e-8:
            std = 1.0

        return (
            mean,
            std,
        )

    # ---------------------------------------------------------------------

    def _compute_lab_statistics(
        self,
        image: np.ndarray,
    ) -> dict[str, tuple[float, float]]:
        """
        Compute LAB channel statistics.

        Returns
        -------
        dict
            Mean and standard deviation
            for L, A and B channels.
        """

        image = self._prepare_image(
            image,
        )

        lab = self._bgr_to_lab(
            image,
        )

        l_channel, a_channel, b_channel = (
            cv2.split(lab)
        )

        statistics = {

            "L": self._compute_mean_std(
                l_channel,
            ),

            "A": self._compute_mean_std(
                a_channel,
            ),

            "B": self._compute_mean_std(
                b_channel,
            ),

        }

        return statistics
    # ---------------------------------------------------------------------

    def _compute_reference_statistics(
        self,
    ) -> dict[str, tuple[float, float]]:
        """
        Compute statistics for the reference image.
        """

        reference = self._load_reference_image()

        statistics = self._compute_lab_statistics(
            reference,
        )

        if (
            self.config.stain_normalization
            .cache_reference_statistics
        ):
            self.config.stain_normalization.reference_statistics = (
                self._copy_statistics(
                    statistics,
                )
            )

        return statistics

    # ---------------------------------------------------------------------

    def _get_reference_statistics(
        self,
    ) -> dict[str, tuple[float, float]]:
        """
        Return cached reference statistics if available,
        otherwise compute them.
        """

        config = (
            self.config.stain_normalization
        )

        if (
            config.force_recompute_statistics
        ):
            return (
                self._compute_reference_statistics()
            )

        cached = (
            config.reference_statistics
        )

        if (
            config.cache_reference_statistics
            and cached is not None
        ):
            return self._copy_statistics(
                cached,
            )

        return (
            self._compute_reference_statistics()
        )

    # ---------------------------------------------------------------------

    @staticmethod
    def _split_lab_channels(
        lab_image: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Split a LAB image into its channels.
        """

        l_channel, a_channel, b_channel = (
            cv2.split(
                lab_image,
            )
        )

        return (
            l_channel.astype(
                np.float32,
            ),
            a_channel.astype(
                np.float32,
            ),
            b_channel.astype(
                np.float32,
            ),
        )

    # ---------------------------------------------------------------------

    @staticmethod
    def _merge_lab_channels(
        l_channel: np.ndarray,
        a_channel: np.ndarray,
        b_channel: np.ndarray,
    ) -> np.ndarray:
        """
        Merge LAB channels into one image.
        """

        return cv2.merge(
            (
                l_channel.astype(
                    np.uint8,
                ),
                a_channel.astype(
                    np.uint8,
                ),
                b_channel.astype(
                    np.uint8,
                ),
            )
        )

    # ---------------------------------------------------------------------

    @staticmethod
    def _copy_statistics(
        statistics: dict[
            str,
            tuple[
                float,
                float,
            ],
        ],
    ) -> dict[
        str,
        tuple[
            float,
            float,
        ],
    ]:
        """
        Create a deep copy of channel statistics.
        """

        copied: dict[
            str,
            tuple[
                float,
                float,
            ],
        ] = {}

        for (
            channel,
            values,
        ) in statistics.items():

            copied[channel] = (
                float(values[0]),
                float(values[1]),
            )

        return copied

    # ---------------------------------------------------------------------

    def _clear_reference_cache(
        self,
    ) -> None:
        """
        Clear cached reference statistics.
        """

        self.config.stain_normalization.reference_statistics = (
            None
        )

    # ---------------------------------------------------------------------

    def _reference_cache_available(
        self,
    ) -> bool:
        """
        Check whether cached statistics exist.
        """

        config = (
            self.config.stain_normalization
        )

        return (
            config.cache_reference_statistics
            and config.reference_statistics
            is not None
        )
# BloodCellAI - stain_normalization.py
# Section 3A-2 and Section 3B

    def _normalize_single_channel(
        self,
        channel,
        source_mean,
        source_std,
        reference_mean,
        reference_std,
    ):
        channel = channel.astype(np.float32)
        normalized = ((channel - source_mean) * (reference_std / max(source_std, 1e-8))) + reference_mean
        return np.clip(normalized, 0, 255).astype(np.float32)

    def _normalize_lab_channels(
        self,
        l_channel,
        a_channel,
        b_channel,
        source_statistics,
        reference_statistics,
    ):
        l = self._normalize_single_channel(l_channel, source_statistics["L"][0], source_statistics["L"][1], reference_statistics["L"][0], reference_statistics["L"][1])
        a = self._normalize_single_channel(a_channel, source_statistics["A"][0], source_statistics["A"][1], reference_statistics["A"][0], reference_statistics["A"][1])
        b = self._normalize_single_channel(b_channel, source_statistics["B"][0], source_statistics["B"][1], reference_statistics["B"][0], reference_statistics["B"][1])
        return l, a, b

    def _clip_image(self, image):
        return np.clip(image, 0, 255).astype(np.uint8)

    def _reconstruct_lab_image(self, l_channel, a_channel, b_channel):
        lab = self._merge_lab_channels(
            self._clip_image(l_channel),
            self._clip_image(a_channel),
            self._clip_image(b_channel),
        )
        return self._lab_to_bgr(lab)

    def _finalize_normalized_image(self, image):
        image = self._clip_image(image)
        return self._prepare_image(image)

    def _reinhard_normalization(self, image):
        image = self._prepare_image(image)
        reference_statistics = self._get_reference_statistics()
        lab = self._bgr_to_lab(image)
        l_channel, a_channel, b_channel = self._split_lab_channels(lab)
        source_statistics = self._compute_lab_statistics(image)
        l_channel, a_channel, b_channel = self._normalize_lab_channels(
            l_channel,
            a_channel,
            b_channel,
            source_statistics,
            reference_statistics,
        )
        normalized = self._reconstruct_lab_image(
            l_channel,
            a_channel,
            b_channel,
        )
        return self._finalize_normalized_image(normalized)

    def apply_reinhard(self, image):
        return self._reinhard_normalization(image)
    # =====================================================================
    # Section 4A
    # Macenko Stain Normalization
    # Optical Density Utilities
    # =====================================================================

    def _rgb_to_optical_density(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Convert RGB image to Optical Density (OD) space.
        """

        image = image.astype(
            np.float32,
        )

        image[image == 0.0] = 1.0

        optical_density = -np.log(
            image / 255.0
        )

        return optical_density

    # ---------------------------------------------------------------------

    def _optical_density_to_rgb(
        self,
        optical_density: np.ndarray,
    ) -> np.ndarray:
        """
        Convert Optical Density image back to RGB.
        """

        rgb = (
            np.exp(
                -optical_density
            ) * 255.0
        )

        rgb = np.clip(
            rgb,
            0.0,
            255.0,
        )

        return rgb.astype(
            np.uint8,
        )

    # ---------------------------------------------------------------------

    def _flatten_optical_density(
        self,
        optical_density: np.ndarray,
    ) -> np.ndarray:
        """
        Convert H×W×3 OD image into N×3 matrix.
        """

        return optical_density.reshape(
            (
                -1,
                3,
            )
        )

    # ---------------------------------------------------------------------

    def _restore_optical_density(
        self,
        optical_density: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Restore flattened OD matrix back to image.
        """

        height, width = shape

        return optical_density.reshape(
            (
                height,
                width,
                3,
            )
        )

    # ---------------------------------------------------------------------

    def _compute_tissue_mask(
        self,
        optical_density: np.ndarray,
    ) -> np.ndarray:
        """
        Compute tissue mask using OD threshold.
        """

        threshold = (
            self.config
            .stain_normalization
            .optical_density_threshold
        )

        mask = np.any(
            optical_density > threshold,
            axis=2,
        )

        return mask

    # ---------------------------------------------------------------------

    def _apply_tissue_mask(
        self,
        optical_density: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Keep only tissue pixels.
        """

        return optical_density[
            mask
        ]

    # ---------------------------------------------------------------------

    def _remove_invalid_pixels(
        self,
        optical_density: np.ndarray,
    ) -> np.ndarray:
        """
        Remove NaN and Inf values.
        """

        valid = np.all(
            np.isfinite(
                optical_density
            ),
            axis=1,
        )

        return optical_density[
            valid
        ]

    # ---------------------------------------------------------------------

    def _prepare_optical_density(
        self,
        image: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Prepare optical density matrix for Macenko.
        """

        image = self._prepare_image(
            image,
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        optical_density = (
            self._rgb_to_optical_density(
                rgb,
            )
        )

        mask = self._compute_tissue_mask(
            optical_density,
        )

        optical_density = (
            self._apply_tissue_mask(
                optical_density,
                mask,
            )
        )

        optical_density = (
            self._remove_invalid_pixels(
                optical_density,
            )
        )

        return (
            optical_density,
            mask,
        )

    # ---------------------------------------------------------------------

    def _validate_optical_density(
        self,
        optical_density: np.ndarray,
    ) -> None:
        """
        Validate OD matrix.
        """

        if optical_density.size == 0:

            raise ValueError(
                "Optical density matrix is empty."
            )

        if optical_density.ndim != 2:

            raise ValueError(
                "Optical density must be N×3."
            )

        if optical_density.shape[1] != 3:

            raise ValueError(
                "Optical density matrix must "
                "contain three columns."
            )

        if np.isnan(
            optical_density
        ).any():

            raise ValueError(
                "NaN values found in optical density."
            )

        if np.isinf(
            optical_density
        ).any():

            raise ValueError(
                "Infinite values found in optical density."
            )

    # ---------------------------------------------------------------------

    def _prepare_macenko_input(
        self,
        image: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[int, int],
    ]:
        """
        Prepare image for Macenko normalization.
        """

        image = self._prepare_image(
            image,
        )

        height, width = image.shape[:2]

        optical_density, mask = (
            self._prepare_optical_density(
                image,
            )
        )

        self._validate_optical_density(
            optical_density,
        )

        return (
            optical_density,
            mask,
            (
                height,
                width,
            ),
        )

    # =====================================================================
    # Section 4B
    # Macenko Stain Normalization
    # SVD Decomposition and Eigenvectors
    # =====================================================================

    def _compute_covariance_matrix(
        self,
        optical_density: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the 3x3 covariance matrix of tissue-masked OD pixels.

        Parameters
        ----------
        optical_density : np.ndarray
            N x 3 tissue-masked OD matrix.

        Returns
        -------
        np.ndarray
            3x3 covariance matrix.
        """

        covariance = np.cov(
            optical_density,
            rowvar=False,
        )

        return covariance

    # ---------------------------------------------------------------------

    def _svd_decomposition(
        self,
        covariance: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Perform SVD on the OD covariance matrix.

        The covariance matrix is symmetric and positive
        semi-definite, so its singular values equal its
        eigenvalues and its left singular vectors equal its
        eigenvectors -- `np.linalg.svd` returns them already
        sorted by descending singular value, which is what the
        rest of the Macenko pipeline needs.

        Parameters
        ----------
        covariance : np.ndarray
            3x3 covariance matrix.

        Returns
        -------
        tuple
            (left_singular_vectors, singular_values), where
            left_singular_vectors is 3x3 (columns are the
            components, in descending order of singular value).
        """

        left_singular_vectors, singular_values, _ = np.linalg.svd(
            covariance,
        )

        return (
            left_singular_vectors,
            singular_values,
        )

    # ---------------------------------------------------------------------

    def _select_principal_eigenvectors(
        self,
        left_singular_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Select the two eigenvectors corresponding to the two
        largest eigenvalues.

        Parameters
        ----------
        left_singular_vectors : np.ndarray
            3x3 matrix returned by `_svd_decomposition`, columns
            already sorted by descending singular value.

        Returns
        -------
        np.ndarray
            3x2 matrix -- the plane spanned by the two principal
            stain directions.
        """

        return left_singular_vectors[:, :2]

    # =====================================================================
    # Section 4C
    # Macenko Stain Normalization
    # Stain Vector Estimation
    # =====================================================================

    def _project_onto_plane(
        self,
        optical_density: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> np.ndarray:
        """
        Project tissue OD pixels onto the plane spanned by the two
        principal eigenvectors.

        Parameters
        ----------
        optical_density : np.ndarray
            N x 3 tissue-masked OD matrix.

        eigenvectors : np.ndarray
            3x2 plane basis from `_select_principal_eigenvectors`.

        Returns
        -------
        np.ndarray
            N x 2 projected coordinates.
        """

        projection = optical_density @ eigenvectors

        return projection

    # ---------------------------------------------------------------------

    def _compute_projection_angles(
        self,
        projection: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the angle of every projected OD point around the
        plane origin.

        Parameters
        ----------
        projection : np.ndarray
            N x 2 projected OD coordinates.

        Returns
        -------
        np.ndarray
            N-length array of angles in radians.
        """

        angles = np.arctan2(
            projection[:, 1],
            projection[:, 0],
        )

        return angles

    # ---------------------------------------------------------------------

    def _angle_to_stain_vector(
        self,
        angle: float,
        eigenvectors: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a single angle on the projection plane back into a
        3-dimensional (unit) OD stain vector.

        Parameters
        ----------
        angle : float
            Angle in radians.

        eigenvectors : np.ndarray
            3x2 plane basis from `_select_principal_eigenvectors`.

        Returns
        -------
        np.ndarray
            3-length unit stain vector.
        """

        direction = np.array(
            [
                np.cos(angle),
                np.sin(angle),
            ]
        )

        stain_vector = eigenvectors @ direction

        norm = np.linalg.norm(stain_vector)

        if norm < 1e-8:
            norm = 1.0

        return stain_vector / norm

    # ---------------------------------------------------------------------

    def _order_stain_vectors(
        self,
        stain_vector_a: np.ndarray,
        stain_vector_b: np.ndarray,
    ) -> np.ndarray:
        """
        Order two stain vectors so that Hematoxylin comes first and
        Eosin comes second, matching the standard H&E convention
        used by every other Macenko implementation.

        Hematoxylin absorbs more strongly in the blue channel than
        Eosin does, so (by the common convention used in the
        original Macenko method) the vector with the larger
        (R - B) OD-channel difference is treated as Eosin, and the
        other as Hematoxylin.

        Parameters
        ----------
        stain_vector_a, stain_vector_b : np.ndarray
            Two 3-length unit stain vectors (order not yet known).

        Returns
        -------
        np.ndarray
            3x2 matrix, columns ordered [Hematoxylin, Eosin].
        """

        if stain_vector_a[0] > stain_vector_b[0]:
            hematoxylin, eosin = stain_vector_b, stain_vector_a
        else:
            hematoxylin, eosin = stain_vector_a, stain_vector_b

        stain_vectors = np.stack(
            [
                hematoxylin,
                eosin,
            ],
            axis=1,
        )

        return stain_vectors

    # ---------------------------------------------------------------------

    def _estimate_stain_vectors(
        self,
        optical_density: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the two H&E stain vectors from tissue-masked OD
        pixels, using the robust angle-percentile method described
        in Macenko et al. (2009).

        Rather than taking the absolute min/max angle (which is
        sensitive to noise/outlier pixels), the `alpha`-th and
        `(100 - alpha)`-th percentiles of the angle distribution are
        used -- `self._alpha` (default 1.0) controls this, matching
        the paper's default.

        Parameters
        ----------
        optical_density : np.ndarray
            N x 3 tissue-masked OD matrix.

        eigenvectors : np.ndarray
            3x2 plane basis from `_select_principal_eigenvectors`.

        Returns
        -------
        np.ndarray
            3x2 matrix, columns ordered [Hematoxylin, Eosin].
        """

        projection = self._project_onto_plane(
            optical_density,
            eigenvectors,
        )

        angles = self._compute_projection_angles(
            projection,
        )

        min_angle = np.percentile(
            angles,
            self._alpha,
        )

        max_angle = np.percentile(
            angles,
            100.0 - self._alpha,
        )

        stain_vector_a = self._angle_to_stain_vector(
            min_angle,
            eigenvectors,
        )

        stain_vector_b = self._angle_to_stain_vector(
            max_angle,
            eigenvectors,
        )

        stain_vectors = self._order_stain_vectors(
            stain_vector_a,
            stain_vector_b,
        )

        return stain_vectors

    # =====================================================================
    # Section 4D
    # Macenko Stain Normalization
    # Concentration Estimation
    # =====================================================================

    def _estimate_concentrations(
        self,
        optical_density: np.ndarray,
        stain_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate per-pixel stain concentrations by solving the
        linear system  OD = concentrations @ stain_vectors.T
        in a least-squares sense.

        Parameters
        ----------
        optical_density : np.ndarray
            N x 3 tissue-masked OD matrix.

        stain_vectors : np.ndarray
            3x2 matrix, columns ordered [Hematoxylin, Eosin].

        Returns
        -------
        np.ndarray
            N x 2 matrix of estimated concentrations.
        """

        concentrations, _residuals, _rank, _singular_values = np.linalg.lstsq(
            stain_vectors,
            optical_density.T,
            rcond=None,
        )

        return concentrations.T

    # ---------------------------------------------------------------------

    def _compute_max_concentrations(
        self,
        concentrations: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a robust "maximum" concentration per stain channel,
        used to rescale source concentrations onto the reference
        image's stain intensity range.

        The 99th percentile is used rather than the true maximum so
        a handful of outlier pixels can't dominate the rescaling.

        Parameters
        ----------
        concentrations : np.ndarray
            N x 2 concentration matrix.

        Returns
        -------
        np.ndarray
            2-length array of maximum concentrations.
        """

        max_concentrations = np.percentile(
            concentrations,
            99,
            axis=0,
        )

        max_concentrations[max_concentrations < 1e-8] = 1e-8

        return max_concentrations

    # ---------------------------------------------------------------------

    def _estimate_stain_and_concentration(
        self,
        image: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[int, int],
    ]:
        """
        Run the full stain/concentration estimation pipeline for a
        single image: OD conversion, tissue masking, SVD, stain
        vector estimation, and concentration estimation.

        Used identically for both the source image and the
        reference image, since Macenko normalization needs both
        estimated the same way.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image.

        Returns
        -------
        tuple
            (stain_vectors, concentrations, max_concentrations,
            tissue_mask, (height, width))
        """

        optical_density, mask, shape = self._prepare_macenko_input(
            image,
        )

        covariance = self._compute_covariance_matrix(
            optical_density,
        )

        left_singular_vectors, _singular_values = self._svd_decomposition(
            covariance,
        )

        eigenvectors = self._select_principal_eigenvectors(
            left_singular_vectors,
        )

        stain_vectors = self._estimate_stain_vectors(
            optical_density,
            eigenvectors,
        )

        concentrations = self._estimate_concentrations(
            optical_density,
            stain_vectors,
        )

        max_concentrations = self._compute_max_concentrations(
            concentrations,
        )

        return (
            stain_vectors,
            concentrations,
            max_concentrations,
            mask,
            shape,
        )

    # =====================================================================
    # Section 4E
    # Macenko Stain Normalization
    # Image Reconstruction
    # =====================================================================

    def _normalize_concentrations(
        self,
        concentrations: np.ndarray,
        source_max_concentrations: np.ndarray,
        reference_max_concentrations: np.ndarray,
    ) -> np.ndarray:
        """
        Rescale source concentrations onto the reference image's
        stain intensity range.

        Parameters
        ----------
        concentrations : np.ndarray
            N x 2 source concentration matrix.

        source_max_concentrations : np.ndarray
            2-length array, source image's robust max concentrations.

        reference_max_concentrations : np.ndarray
            2-length array, reference image's robust max
            concentrations.

        Returns
        -------
        np.ndarray
            N x 2 rescaled concentration matrix.
        """

        scale = reference_max_concentrations / source_max_concentrations

        normalized = concentrations * scale

        return normalized

    # ---------------------------------------------------------------------

    def _reconstruct_tissue_od(
        self,
        concentrations: np.ndarray,
        stain_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct OD values for tissue pixels from (rescaled)
        concentrations and a set of stain vectors.

        Parameters
        ----------
        concentrations : np.ndarray
            N x 2 concentration matrix.

        stain_vectors : np.ndarray
            3x2 matrix, columns ordered [Hematoxylin, Eosin] --
            normally the *reference* image's stain vectors, so the
            reconstructed image adopts the reference's stain
            appearance.

        Returns
        -------
        np.ndarray
            N x 3 reconstructed OD matrix.
        """

        reconstructed_od = concentrations @ stain_vectors.T

        return reconstructed_od

    # ---------------------------------------------------------------------

    def _reinsert_tissue_pixels(
        self,
        original_image: np.ndarray,
        reconstructed_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Reinsert reconstructed tissue pixels into a copy of the
        original image, leaving background (non-tissue) pixels
        untouched.

        Background pixels are deliberately left as-is rather than
        stain-normalized -- they aren't tissue, so there's no stain
        to normalize, and reconstructing them would risk introducing
        color artifacts in empty slide regions.

        Parameters
        ----------
        original_image : np.ndarray
            Original BGR uint8 image (post `_prepare_image`).

        reconstructed_rgb : np.ndarray
            N_tissue x 3 reconstructed RGB values for tissue pixels
            only (N_tissue == mask.sum()).

        mask : np.ndarray
            H x W boolean tissue mask from `_compute_tissue_mask`.

        Returns
        -------
        np.ndarray
            BGR uint8 image, same shape as `original_image`.
        """

        output_rgb = self._bgr_to_rgb(
            original_image,
        ).copy()

        output_rgb[mask] = reconstructed_rgb

        output_bgr = self._rgb_to_bgr(
            output_rgb,
        )

        return output_bgr

    # ---------------------------------------------------------------------

    def _macenko_normalization(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Full Macenko stain normalization pipeline for one image.

        Estimates stain vectors and concentrations for both the
        source image and the reference image, rescales the source
        concentrations onto the reference's stain intensity range,
        and reconstructs the image using the *reference's* stain
        vectors -- this is what makes the output adopt the
        reference's stain appearance while preserving the source
        image's tissue structure.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 source image.

        Returns
        -------
        np.ndarray
            BGR uint8 stain-normalized image.
        """

        image = self._prepare_image(
            image,
        )

        (
            source_stain_vectors,
            source_concentrations,
            source_max_concentrations,
            mask,
            shape,
        ) = self._estimate_stain_and_concentration(
            image,
        )

        reference_image = self._load_reference_image()

        (
            reference_stain_vectors,
            _reference_concentrations,
            reference_max_concentrations,
            _reference_mask,
            _reference_shape,
        ) = self._estimate_stain_and_concentration(
            reference_image,
        )

        normalized_concentrations = self._normalize_concentrations(
            source_concentrations,
            source_max_concentrations,
            reference_max_concentrations,
        )

        reconstructed_od = self._reconstruct_tissue_od(
            normalized_concentrations,
            reference_stain_vectors,
        )

        reconstructed_rgb = self._optical_density_to_rgb(
            reconstructed_od,
        )

        normalized_image = self._reinsert_tissue_pixels(
            image,
            reconstructed_rgb,
            mask,
        )

        self._last_stain_matrix = reference_stain_vectors
        self._last_concentration_matrix = normalized_concentrations

        self._calculate_channel_statistics(
            normalized_image,
        )

        return normalized_image

    # ---------------------------------------------------------------------

    def apply_macenko(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Public entry point for Macenko stain normalization,
        mirroring `apply_reinhard`.
        """

        return self._macenko_normalization(
            image,
        )

    # =====================================================================
    # Section 5
    # Public API
    # =====================================================================

    def apply(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the configured stain normalization method to an image.

        This is the method `BaseTransform`'s `transform()` wrapper
        calls -- it dispatches to the configured method
        (`self._method`) rather than requiring the caller to know
        which `apply_*` method to use.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image.

        Returns
        -------
        np.ndarray
            BGR uint8 image -- unchanged if normalization is
            disabled or the method is NONE.
        """

        if not self._enabled or self._method is StainNormalizationMethod.NONE:
            return self._prepare_image(image)

        if self._method is StainNormalizationMethod.REINHARD:
            return self.apply_reinhard(image)

        if self._method is StainNormalizationMethod.MACENKO:
            return self.apply_macenko(image)

        if self._method is StainNormalizationMethod.HISTOGRAM:
            raise NotImplementedError(
                "Histogram-matching stain normalization is not "
                "implemented yet -- StainNormalizationMethod.HISTOGRAM "
                "is defined but has no corresponding apply_* method "
                "in this file. Use REINHARD or MACENKO for now."
            )

        raise ValueError(
            f"Unsupported stain normalization method: {self._method}"
        )


TransformRegistry.register(
    "stain_normalization",
    StainNormalizationTransform,
)
