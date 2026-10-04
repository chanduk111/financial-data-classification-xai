"""Explainable AI utilities for model interpretation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def get_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return ranked feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The supplied model does not provide feature importances."
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def prepare_lime_data(
    X: Any,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Prepare feature data and names for local explanations."""
    array = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")

    if len(feature_names) != array.shape[1]:
        raise ValueError(
            "Number of feature names must match the number of features."
        )

    return array, feature_names
