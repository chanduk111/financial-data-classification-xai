"""End-to-end training entry point for the classification project."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .models import build_models, evaluate_model


RANDOM_STATE = 42


def train_and_evaluate(
    X,
    y,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Train all configured models and save the comparison results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    comparison = pd.DataFrame(results).T

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        output_path / "model_comparison_generated.csv"
    )

    return comparison


def save_model(model, path: str) -> None:
    """Save a trained model using Joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    print(
        "Training entry point created. "
        "Load the project dataset and construct X/y before calling "
        "train_and_evaluate()."
    )
