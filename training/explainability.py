"""
==============================================================
BloodCellAI Training — Explainability
==============================================================

File:
    explainability.py

Description
-----------
Two explainability paths, matched to what can and can't actually run
in this project's execution environment:

1. permutation_importance_report() -- REAL, runs here. Works with the
   RandomForest classifier from classifier.py (or any sklearn
   estimator). Verified against real BCCD-derived features.

2. GradCAM -- a complete, standard Grad-CAM implementation for a
   PyTorch CNN, for use in a GPU environment (Kaggle/Colab) once you
   train a real CNN there. This class CANNOT be run or verified in
   this sandbox (no PyTorch is installed here, and none could be
   installed -- no network path to a usable wheel and insufficient
   disk space, confirmed directly). It is included because Grad-CAM
   is standard/expected for a CNN-based blood cell detector, and is
   written carefully against the well-established Grad-CAM algorithm,
   but it is explicitly NOT claimed to be tested here -- only the
   permutation-importance path above carries that claim.

Author:
    BloodCellAI Project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


# =============================================================================
# Permutation Importance (real, sklearn-based -- runs in this environment)
# =============================================================================

@dataclass
class FeatureImportanceReport:

    feature_names: list = field(default_factory=list)
    importances_mean: list = field(default_factory=list)
    importances_std: list = field(default_factory=list)

    def ranked(self) -> List[tuple]:
        """
        Return (feature_name, mean_importance, std) sorted by
        importance, most important first.
        """

        combined = list(
            zip(self.feature_names, self.importances_mean, self.importances_std)
        )

        return sorted(combined, key=lambda row: -row[1])

    def summary_text(self, top_n: int = 10) -> str:

        lines = [
            "=" * 70,
            "Feature Importance Report (permutation importance)",
            "=" * 70,
        ]

        for name, mean, std in self.ranked()[:top_n]:
            lines.append(f"  {name:20} {mean:+.4f}  (± {std:.4f})")

        return "\n".join(lines)

    def to_dict(self) -> dict:

        return {
            "feature_names": self.feature_names,
            "importances_mean": self.importances_mean,
            "importances_std": self.importances_std,
        }


def permutation_importance_report(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> FeatureImportanceReport:
    """
    Compute permutation feature importance: how much does shuffling
    one feature column degrade the model's accuracy? A larger drop
    means the model relies on that feature more.

    This is a genuinely model-agnostic, real explainability method --
    it works with the RandomForest from classifier.py (or any fitted
    sklearn estimator), and only needs the already-trained model plus
    a held-out (X, y) set, both of which this pipeline already
    produces.
    """

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    return FeatureImportanceReport(
        feature_names=list(feature_names),
        importances_mean=result.importances_mean.tolist(),
        importances_std=result.importances_std.tolist(),
    )


# =============================================================================
# Grad-CAM (PyTorch, GPU environment only -- NOT runnable in this sandbox)
# =============================================================================

class GradCAM:
    """
    Standard Grad-CAM (Selvaraju et al., 2017) for a PyTorch CNN.

    IMPORTANT: this class requires `torch` and a trained CNN, neither
    of which are usable in this project's sandbox (confirmed: no
    network path to a PyTorch wheel, and insufficient disk space to
    install one even if there were). It is written for use in a GPU
    environment such as Kaggle or Colab, following the guidance given
    earlier in this project for where actual model training belongs.
    It has NOT been executed or verified anywhere in this project --
    unlike everything else in this codebase, which has been. Treat it
    as a correctly-written starting point, not a tested component.

    Usage (in a GPU environment, after training a CNN there)
    ----------------------------------------------------------
        cam = GradCAM(model, target_layer=model.layer4[-1])
        heatmap = cam.generate(input_tensor, class_index=predicted_class)
        overlay = cam.overlay_on_image(heatmap, original_bgr_image)
    """

    def __init__(self, model, target_layer):

        import torch  # deferred import -- only needed if this class is used

        self.torch = torch
        self.model = model
        self.target_layer = target_layer

        self._activations = None
        self._gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_index=None):
        """
        Compute a Grad-CAM heatmap for one input tensor.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, C, H, W), already normalized as the model expects.

        class_index : int, optional
            Which class's CAM to compute. Defaults to the model's own
            top prediction.

        Returns
        -------
        np.ndarray
            2D heatmap, values in [0, 1], same spatial size as the
            target layer's feature map (resize to the input image
            size before overlaying).
        """

        torch = self.torch

        self.model.eval()

        output = self.model(input_tensor)

        if class_index is None:
            class_index = int(output.argmax(dim=1).item())

        self.model.zero_grad()

        score = output[0, class_index]
        score.backward()

        gradients = self._gradients[0]
        activations = self._activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

    @staticmethod
    def overlay_on_image(heatmap: np.ndarray, image_bgr: np.ndarray, alpha: float = 0.4):
        """
        Resize a Grad-CAM heatmap to an image's size and overlay it
        as a color heatmap. Uses OpenCV only (no torch needed for
        this step), so it can be tested independently once you have
        any heatmap array of the right shape.
        """

        import cv2

        height, width = image_bgr.shape[:2]

        resized = cv2.resize(heatmap, (width, height))
        colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        return cv2.addWeighted(colored, alpha, image_bgr, 1 - alpha, 0)
