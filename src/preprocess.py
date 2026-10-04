"""Data preprocessing and feature engineering utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the input dataframe."""
    data = df.copy()
    data = data.drop_duplicates()
    data = data.dropna(how="all")
    return data


def create_tfidf_features(
    text: pd.Series,
    max_features: int = 5000,
) -> tuple:
    """Create TF-IDF features from a text series."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
    )
    features = vectorizer.fit_transform(text.fillna("").astype(str))
    return features, vectorizer
