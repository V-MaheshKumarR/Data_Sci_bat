"""
soundml/backend/ml/train.py
────────────────────────────
Complete model training entrypoint.

Workflow:
  1. Load / generate dataset
  2. Preprocess: encode labels, train/test split
  3. Train all three models with hyperparameter tuning
  4. Evaluate on held-out test set
  5. Select best model
  6. Register with ModelRegistry
  7. Auto-promote if accuracy exceeds threshold

Run:
  python -m backend.ml.train
  python -m backend.ml.train --models knn,random_forest --no-tune
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.data.dataset import build_full_dataset, FEATURES
from backend.data.preprocessing import GenreLabelEncoder, validate_features
from backend.ml.pipeline import MODEL_DEFINITIONS, tune_pipeline, build_pipeline
from backend.ml.evaluate import evaluate_model, print_evaluation_report
from backend.ml.registry import ModelRegistry
from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger("soundml.train")


def prepare_data(
    df: pd.DataFrame,
    label_encoder: GenreLabelEncoder,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and label vectors for training.

    Args:
        df:            Full dataset DataFrame
        label_encoder: Fitted GenreLabelEncoder
        test_size:     Fraction of data reserved for testing
        seed:          Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURES].values.astype(np.float64)
    y = label_encoder.encode(df["genre"].values)

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,  # Preserve class proportions in both splits
    )


def train(
    model_keys: list[str] | None = None,
    tune: bool = True,
    n_synthetic_per_genre: int = 50,
    auto_promote: bool = True,
) -> dict:
    """
    Full training pipeline: data → preprocessing → training → evaluation → registration.

    Args:
        model_keys:           Models to train. None = all models.
        tune:                 Run GridSearchCV. False = use defaults (faster).
        n_synthetic_per_genre: Synthetic data augmentation count.
        auto_promote:         Promote best model if accuracy > threshold.

    Returns:
        Summary dict with results for all trained models.
    """
    start_time = time.time()
    model_keys = model_keys or list(MODEL_DEFINITIONS.keys())
    log.info("Starting training", models=model_keys, tune=tune)

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    log.info("Loading dataset...")
    if settings.DATASET_PATH.exists():
        df = pd.read_csv(settings.DATASET_PATH)
        log.info(f"Loaded existing dataset: {len(df)} rows")
    else:
        df = build_full_dataset(
            n_synthetic_per_genre=n_synthetic_per_genre,
            save_path=settings.DATASET_PATH,
        )
        log.info(f"Generated new dataset: {len(df)} rows")

    # ── 2. Label encoding & split ──────────────────────────────────────────
    label_enc = GenreLabelEncoder()
    X_train, X_test, y_train, y_test = prepare_data(
        df, label_enc,
        test_size=settings.TEST_SIZE,
        seed=settings.RANDOM_STATE,
    )
    log.info(
        "Data split complete",
        train_size=len(X_train),
        test_size=len(X_test),
        classes=label_enc.classes,
    )

    # ── 3. Train each model ────────────────────────────────────────────────
    registry = ModelRegistry()
    results = {}

    for model_key in model_keys:
        log.info(f"Training {model_key}...")
        t0 = time.time()

        if tune:
            pipeline, tune_info = tune_pipeline(
                model_key, X_train, y_train,
                cv_folds=settings.CV_FOLDS,
            )
        else:
            pipeline = build_pipeline(model_key)
            pipeline.fit(X_train, y_train)
            tune_info = {"best_params": {}, "best_cv_accuracy": None}

        train_time = round(time.time() - t0, 2)
        log.info(f"{model_key} trained in {train_time}s", **tune_info)

        # ── 4. Evaluate ────────────────────────────────────────────────────
        eval_metrics = evaluate_model(pipeline, X_test, y_test, label_enc)
        print_evaluation_report(model_key, eval_metrics)

        # ── 5. Register ────────────────────────────────────────────────────
        version = registry.register(
            model_key=model_key,
            pipeline=pipeline,
            label_encoder=label_enc,
            metrics=eval_metrics,
            training_info={
                "n_train": int(len(X_train)),
                "n_test":  int(len(X_test)),
                "tune":    tune,
                "train_time_seconds": train_time,
                **tune_info,
            },
            dataset_info={
                "total_rows": int(len(df)),
                "n_synthetic_per_genre": n_synthetic_per_genre,
                "genre_counts": df["genre"].value_counts().to_dict(),
            },
        )
        log.info(f"Registered {model_key} as version {version}")
        results[model_key] = {"version": version, "metrics": eval_metrics}

    # ── 6. Select & auto-promote best model ───────────────────────────────
    best_key = max(results, key=lambda k: results[k]["metrics"]["accuracy"])
    best_acc = results[best_key]["metrics"]["accuracy"]
    best_version = results[best_key]["version"]

    log.info(f"Best model: {best_key} (version={best_version}, accuracy={best_acc:.4f})")

    if auto_promote and best_acc >= settings.AUTO_PROMOTE_THRESHOLD:
        registry.promote(best_version)
        log.info(f"Auto-promoted version {best_version} to production")
    else:
        log.info(
            "Auto-promotion skipped",
            reason="threshold not met" if best_acc < settings.AUTO_PROMOTE_THRESHOLD else "disabled",
            threshold=settings.AUTO_PROMOTE_THRESHOLD,
            best_accuracy=best_acc,
        )

    total_time = round(time.time() - start_time, 2)
    log.info(f"Training complete in {total_time}s")

    return {
        "models": results,
        "best_model": best_key,
        "best_version": best_version,
        "best_accuracy": best_acc,
        "total_time_seconds": total_time,
    }


# ── CLI entrypoint ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoundML Model Training")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model keys to train (default: all). E.g. knn,random_forest"
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip GridSearchCV (faster, uses default hyperparams)"
    )
    parser.add_argument(
        "--synthetic", type=int, default=50,
        help="Synthetic samples per genre (default: 50)"
    )
    parser.add_argument(
        "--no-promote", action="store_true",
        help="Disable auto-promotion to production"
    )
    args = parser.parse_args()

    model_keys = [m.strip() for m in args.models.split(",")] if args.models else None

    summary = train(
        model_keys=model_keys,
        tune=not args.no_tune,
        n_synthetic_per_genre=args.synthetic,
        auto_promote=not args.no_promote,
    )

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model, info in summary["models"].items():
        acc = info["metrics"]["accuracy"]
        f1  = info["metrics"]["macro_f1"]
        print(f"  {model:<20} acc={acc:.4f}  f1={f1:.4f}  v={info['version']}")
    print(f"\n  Best: {summary['best_model']} (v{summary['best_version']}) → acc={summary['best_accuracy']:.4f}")
    print(f"  Total time: {summary['total_time_seconds']}s")
    print("="*60)
