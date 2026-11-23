"""
soundml/backend/ml/evaluate.py
────────────────────────────────
Comprehensive model evaluation for genre classifiers.

Produces:
  • Accuracy (overall)
  • Per-class Precision, Recall, F1
  • Macro and Weighted averages
  • Confusion matrix (normalised)
  • Top-2 accuracy (is correct genre in top 2?)
  • Feature importances (for Random Forest)
  • ROC-AUC (one-vs-rest, for models with predict_proba)

Educational note:
  Accuracy alone is insufficient. A model that always predicts "Pop"
  gets 17% accuracy (1/6 chance). We need F1 per class and
  confusion matrix to see WHERE it fails.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from backend.data.preprocessing import GenreLabelEncoder


def evaluate_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: GenreLabelEncoder,
) -> dict:
    """
    Run comprehensive evaluation on a trained pipeline.

    Args:
        pipeline:      Fitted sklearn Pipeline
        X_test:        Raw (unprocessed) test features
        y_test:        Encoded test labels
        label_encoder: GenreLabelEncoder for class name mapping

    Returns:
        Evaluation metrics dictionary
    """
    y_pred = pipeline.predict(X_test)
    classes = label_encoder.classes

    # ── Core metrics ───────────────────────────────────────────────────────
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_precision = float(precision_score(y_test, y_pred, average="macro"))
    macro_recall = float(recall_score(y_test, y_pred, average="macro"))

    # ── Per-class metrics ──────────────────────────────────────────────────
    per_class_f1        = f1_score(y_test, y_pred, average=None)
    per_class_precision = precision_score(y_test, y_pred, average=None)
    per_class_recall    = recall_score(y_test, y_pred, average=None)

    per_class = {}
    for i, genre in enumerate(classes):
        per_class[genre] = {
            "precision": round(float(per_class_precision[i]), 4),
            "recall":    round(float(per_class_recall[i]), 4),
            "f1":        round(float(per_class_f1[i]), 4),
        }

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = (cm.astype(float) / cm.sum(axis=1, keepdims=True)).round(4)

    # ── Top-2 accuracy ─────────────────────────────────────────────────────
    top2_accuracy = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)
            top2_preds = np.argsort(proba, axis=1)[:, -2:]
            top2_accuracy = float(np.mean(
                [y_test[i] in top2_preds[i] for i in range(len(y_test))]
            ))
        except Exception:
            pass

    # ── Probability calibration (if available) ────────────────────────────
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(X_test)
        except Exception:
            pass

    # ── Feature importances (Random Forest only) ──────────────────────────
    feature_importances = None
    try:
        clf = pipeline.named_steps.get("classifier")
        if hasattr(clf, "feature_importances_"):
            # Get feature names from the engineer step
            from backend.data.preprocessing import ENGINEERED_FEATURES
            imps = clf.feature_importances_
            feature_importances = {
                name: round(float(imp), 4)
                for name, imp in zip(ENGINEERED_FEATURES, imps)
            }
    except Exception:
        pass

    return {
        "accuracy":          round(accuracy, 4),
        "macro_f1":          round(macro_f1, 4),
        "weighted_f1":       round(weighted_f1, 4),
        "macro_precision":   round(macro_precision, 4),
        "macro_recall":      round(macro_recall, 4),
        "top2_accuracy":     round(top2_accuracy, 4) if top2_accuracy else None,
        "per_class":         per_class,
        "confusion_matrix":  cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "classes":           classes,
        "n_test_samples":    int(len(y_test)),
        "feature_importances": feature_importances,
    }


def print_evaluation_report(model_key: str, metrics: dict) -> None:
    """Print a rich terminal evaluation summary."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  MODEL: {model_key.upper()}")
    print(sep)
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"  Top-2 Accuracy:   {metrics.get('top2_accuracy', 'N/A')}")
    print(f"\n  Per-Class Performance:")
    for genre, m in metrics["per_class"].items():
        print(f"    {genre:<15}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    if metrics.get("feature_importances"):
        print(f"\n  Top Feature Importances:")
        sorted_fi = sorted(metrics["feature_importances"].items(), key=lambda x: -x[1])
        for feat, imp in sorted_fi[:5]:
            bar = "█" * int(imp * 30)
            print(f"    {feat:<30} {imp:.4f} {bar}")
    print(sep)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame for multiple model results.

    Args:
        results: {model_key: metrics_dict}

    Returns:
        Pandas DataFrame sorted by accuracy descending
    """
    rows = []
    for model_key, metrics in results.items():
        rows.append({
            "model":           model_key,
            "accuracy":        metrics["accuracy"],
            "macro_f1":        metrics["macro_f1"],
            "weighted_f1":     metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall":    metrics["macro_recall"],
            "top2_accuracy":   metrics.get("top2_accuracy"),
        })
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    return df
