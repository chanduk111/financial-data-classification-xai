"""Machine learning model training and evaluation utilities."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def build_models() -> Dict[str, object]:
    """Create the classification models used in the project."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained classification model."""
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Convert model evaluation results into a comparison table."""
    return pd.DataFrame(results).T.sort_values(
        by="f1_score",
        ascending=False,
    )
