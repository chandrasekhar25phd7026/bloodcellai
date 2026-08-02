"""
==============================================================
BloodCellAI Training — Classifier Training & Evaluation
==============================================================

File:
    classifier.py

Description
-----------
Trains and evaluates a classical ML classifier (RandomForest, via
scikit-learn) on engineered features from patch_features.py, and
produces a full evaluation report (accuracy, precision, recall, F1,
confusion matrix).

Author:
    BloodCellAI Project
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """
    Full evaluation result for a trained classifier.
    """

    accuracy: float = 0.0
    precision_per_class: dict = field(default_factory=dict)
    recall_per_class: dict = field(default_factory=dict)
    f1_per_class: dict = field(default_factory=dict)
    macro_f1: float = 0.0
    confusion_matrix: list = field(default_factory=list)
    class_names: list = field(default_factory=list)
    support_per_class: dict = field(default_factory=dict)
    text_report: str = ""

    def to_dict(self) -> dict:

        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class,
            "support_per_class": self.support_per_class,
            "confusion_matrix": self.confusion_matrix,
            "class_names": self.class_names,
        }

    def summary_text(self) -> str:

        lines = [
            "=" * 70,
            "Evaluation Report",
            "=" * 70,
            f"Accuracy : {self.accuracy:.4f}",
            f"Macro F1 : {self.macro_f1:.4f}",
            "",
            self.text_report,
        ]

        return "\n".join(lines)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
) -> RandomForestClassifier:
    """
    Train a RandomForest classifier.

    `class_weight="balanced"` matters here specifically because real
    blood cell datasets (confirmed on real BCCD data earlier in this
    project) are heavily imbalanced -- e.g. BCCD is ~85% RBC, ~7.6%
    WBC, ~7.4% Platelet -- an unweighted classifier would just learn
    to always predict the majority class and still score deceptively
    well on raw accuracy.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
) -> EvaluationReport:
    """
    Evaluate a trained classifier and produce a full report.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    labels = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )

    macro_f1 = float(np.mean(f1))

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    text_report = classification_report(
        y_test, y_pred, labels=labels, target_names=class_names, zero_division=0
    )

    report = EvaluationReport(
        accuracy=float(accuracy),
        precision_per_class={name: float(p) for name, p in zip(class_names, precision)},
        recall_per_class={name: float(r) for name, r in zip(class_names, recall)},
        f1_per_class={name: float(f) for name, f in zip(class_names, f1)},
        macro_f1=macro_f1,
        confusion_matrix=cm.tolist(),
        class_names=class_names,
        support_per_class={name: int(s) for name, s in zip(class_names, support)},
        text_report=text_report,
    )

    return report


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    test_size: float = 0.25,
    random_state: int = 42,
):
    """
    Convenience wrapper: split, train, evaluate in one call.

    Returns
    -------
    tuple
        (model, EvaluationReport)
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = train_classifier(X_train, y_train, random_state=random_state)

    report = evaluate_classifier(model, X_test, y_test, class_names)

    return model, report
