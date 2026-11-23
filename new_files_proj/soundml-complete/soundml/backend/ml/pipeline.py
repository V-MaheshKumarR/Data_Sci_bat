"""
soundml/backend/ml/pipeline.py
────────────────────────────────
Full sklearn Pipeline definitions for genre classification.

Three classifiers are benchmarked:
  1. KNN   — K-Nearest Neighbors (baseline, highly interpretable)
  2. RF    — Random Forest (ensemble, high accuracy)
  3. SVM   — Support Vector Machine (strong on high-dim data)

Each is wrapped in a full Pipeline:
  preprocessor → model

The best model (by CV accuracy) is selected and registered.

Educational note:
  Using sklearn Pipelines ensures that:
  - Preprocessing is fit ONLY on training data (no data leakage)
  - The entire transform → predict chain is a single .predict() call
  - The pipeline is serialisable as a single artifact
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from backend.data.preprocessing import build_preprocessor

# ── Model definitions with hyperparameter grids ────────────────────────────

MODEL_DEFINITIONS = {
    "knn": {
        "name": "K-Nearest Neighbors",
        "short": "KNN",
        "classifier": KNeighborsClassifier(),
        "param_grid": {
            # Prefix 'classifier__' routes params to the Pipeline step
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights":     ["uniform", "distance"],
            "classifier__metric":      ["euclidean", "manhattan"],
        },
        "description": (
            "Instance-based learning. No explicit training phase; "
            "classifies by majority vote among K nearest training examples. "
            "Highly interpretable — ideal for teaching."
        ),
    },
    "random_forest": {
        "name": "Random Forest",
        "short": "RF",
        "classifier": RandomForestClassifier(random_state=42),
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth":    [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
        },
        "description": (
            "Ensemble of decision trees trained on bootstrap samples. "
            "Robust to overfitting, provides feature importances, "
            "and typically achieves best accuracy."
        ),
    },
    "svm": {
        "name": "Support Vector Machine",
        "short": "SVM",
        "classifier": SVC(probability=True, random_state=42),
        "param_grid": {
            "classifier__C":      [0.1, 1.0, 10.0],
            "classifier__kernel": ["rbf", "poly"],
            "classifier__gamma":  ["scale", "auto"],
        },
        "description": (
            "Finds maximum-margin hyperplanes between classes. "
            "Excellent in high-dimensional spaces. "
            "RBF kernel handles non-linear decision boundaries."
        ),
    },
}


def build_pipeline(model_key: str) -> Pipeline:
    """
    Build a full sklearn Pipeline for a named model.

    Structure:
        preprocessor (OutlierClipper → FeatureEngineer → StandardScaler)
        → classifier

    Args:
        model_key: One of 'knn', 'random_forest', 'svm'

    Returns:
        sklearn Pipeline ready for .fit()
    """
    if model_key not in MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model: '{model_key}'. Choose from {list(MODEL_DEFINITIONS)}")

    definition = MODEL_DEFINITIONS[model_key]
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier",   definition["classifier"]),
    ])


def tune_pipeline(
    model_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> tuple[Pipeline, dict]:
    """
    Run GridSearchCV to find the best hyperparameters for a given model.

    Uses StratifiedKFold to ensure balanced genre representation
    in each fold — important when genre counts are unequal.

    Args:
        model_key:  Model identifier
        X_train:    Feature matrix (n_samples, n_features)
        y_train:    Encoded genre labels
        cv_folds:   Number of cross-validation folds
        n_jobs:     Parallel jobs (-1 = use all cores)
        verbose:    GridSearchCV verbosity

    Returns:
        (best_pipeline, best_params_dict)
    """
    definition = MODEL_DEFINITIONS[model_key]
    pipeline = build_pipeline(model_key)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=definition["param_grid"],
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    best_params = {
        k.replace("classifier__", ""): v
        for k, v in grid_search.best_params_.items()
    }

    return grid_search.best_estimator_, {
        "best_params":       best_params,
        "best_cv_accuracy":  float(grid_search.best_score_),
        "cv_results_mean":   float(np.mean(grid_search.cv_results_["mean_test_score"])),
        "cv_results_std":    float(np.mean(grid_search.cv_results_["std_test_score"])),
    }
