"""
soundml/backend/data/preprocessing.py
───────────────────────────────────────
Feature preprocessing and engineering for SoundML.

Stages:
  1. Input validation
  2. Outlier detection (IQR-based)
  3. Feature engineering (composite features)
  4. Scaling (StandardScaler)
  5. Optional dimensionality reduction (PCA — for visualisation)

Educational note:
  This module separates concerns cleanly:
  - Raw data → validated → engineered → scaled
  The sklearn Pipeline wires these steps together in ml/pipeline.py.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

FEATURES = ["danceability", "energy", "valence", "tempo", "acousticness", "speechiness"]
ENGINEERED_FEATURES = FEATURES + [
    "energy_dance_product",   # energy × danceability
    "valence_energy_ratio",   # valence / (energy + ε) — mood per intensity unit
    "acoustic_speech_diff",   # acousticness - speechiness
    "tempo_normalized",       # tempo scaled to [0,1] using domain knowledge
]


# ── Custom Transformers ────────────────────────────────────────────────────

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates composite features that capture genre-specific patterns
    not fully captured by individual features.

    These new features provide the model with "pre-digested" signals:
    • energy_dance_product → high for Electronic, Hip-Hop
    • valence_energy_ratio → high for happy-but-mellow (Country, R&B)
    • acoustic_speech_diff → separates Country (+) from Hip-Hop (-)
    • tempo_normalized     → linear scaling without data leakage
    """

    def fit(self, X, y=None):
        return self  # Stateless transformer

    def transform(self, X):
        # Accept both DataFrame and ndarray
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=FEATURES[:X.shape[1]])
        else:
            df = X.copy()

        eps = 1e-6  # Avoid division by zero

        df["energy_dance_product"] = df["energy"] * df["danceability"]
        df["valence_energy_ratio"] = df["valence"] / (df["energy"] + eps)
        df["acoustic_speech_diff"] = df["acousticness"] - df["speechiness"]
        df["tempo_normalized"] = (df["tempo"] - 60.0) / (200.0 - 60.0)  # Domain-scaled

        return df.values.astype(np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(ENGINEERED_FEATURES)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips feature values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] bounds
    learned during fit. Handles edge cases gracefully.

    Why clipping instead of removal?
    → At inference time we receive single samples; we can't remove rows.
      Clipping keeps the pipeline stateless and safe.
    """

    def __init__(self):
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=FEATURES[:X.shape[1]])
        else:
            df = X[FEATURES].copy() if hasattr(X, 'columns') else pd.DataFrame(X)

        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - 1.5 * iqr).values
        self.upper_ = (q3 + 1.5 * iqr).values
        return self

    def transform(self, X):
        arr = X if isinstance(X, np.ndarray) else X[FEATURES].values
        return np.clip(arr, self.lower_, self.upper_)


# ── Label encoding utilities ───────────────────────────────────────────────

class GenreLabelEncoder:
    """
    Wraps sklearn's LabelEncoder with convenience methods.
    Maintains a stable mapping even if genres appear in different order.
    """
    GENRE_ORDER = ["Pop", "Hip-Hop", "Electronic", "Rock", "R&B", "Country"]

    def __init__(self):
        self._enc = LabelEncoder()
        self._enc.fit(self.GENRE_ORDER)  # Stable ordering

    def encode(self, labels) -> np.ndarray:
        return self._enc.transform(labels)

    def decode(self, indices) -> np.ndarray:
        return self._enc.inverse_transform(indices)

    def decode_one(self, index: int) -> str:
        return self._enc.classes_[index]

    @property
    def classes(self) -> list[str]:
        return list(self._enc.classes_)

    @property
    def n_classes(self) -> int:
        return len(self._enc.classes_)


# ── Preprocessing pipeline builder ────────────────────────────────────────

def build_preprocessor() -> Pipeline:
    """
    Build the sklearn preprocessing pipeline:
      1. OutlierClipper  — Clip outliers using IQR method
      2. FeatureEngineer — Create composite features
      3. StandardScaler  — Zero mean, unit variance scaling

    Returns a fitted-ready Pipeline.
    """
    return Pipeline([
        ("outlier_clip",  OutlierClipper()),
        ("feature_eng",   FeatureEngineer()),
        ("scaler",        StandardScaler()),
    ])


# ── Input validation ───────────────────────────────────────────────────────

def validate_features(sample: dict) -> tuple[bool, str]:
    """
    Validate a raw input sample before preprocessing.

    Returns:
        (is_valid, error_message)
    """
    required = set(FEATURES)
    missing = required - set(sample.keys())
    if missing:
        return False, f"Missing features: {sorted(missing)}"

    bounds = {
        "danceability": (0.0, 1.0),
        "energy":       (0.0, 1.0),
        "valence":      (0.0, 1.0),
        "tempo":        (40.0, 250.0),
        "acousticness": (0.0, 1.0),
        "speechiness":  (0.0, 1.0),
    }
    for feat, (lo, hi) in bounds.items():
        val = sample.get(feat)
        if val is None:
            return False, f"Feature '{feat}' is None"
        if not isinstance(val, (int, float)):
            return False, f"Feature '{feat}' must be numeric, got {type(val)}"
        if not (lo <= val <= hi):
            return False, f"Feature '{feat}' out of range [{lo}, {hi}]: got {val}"

    return True, ""


def sample_to_dataframe(sample: dict) -> pd.DataFrame:
    """Convert a single sample dict to a one-row DataFrame with correct column order."""
    return pd.DataFrame([{f: sample[f] for f in FEATURES}])
