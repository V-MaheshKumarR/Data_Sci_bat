"""
soundml/backend/api/routes/health.py
───────────────────────────────────────
Health check, readiness probe, and metrics endpoints.

GET /health      → Liveness probe (always 200 if server is running)
GET /health/ready → Readiness probe (200 only if model is loaded)
GET /metrics      → Current model performance metrics
"""

from fastapi import APIRouter, Request
from datetime import datetime, timezone
import platform
import sys

from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger("soundml.health")
router = APIRouter()
_start_time = datetime.now(timezone.utc)


@router.get(
    "/health",
    summary="Liveness probe",
    description="Returns 200 if the API process is running. Used by container orchestrators (k8s).",
)
async def health() -> dict:
    uptime = (datetime.now(timezone.utc) - _start_time).total_seconds()
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "uptime_seconds": round(uptime, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Returns 200 only if a production model is loaded and ready to serve.",
)
async def readiness(request: Request) -> dict:
    from backend.api.main import app_state
    pipeline = app_state.get("pipeline")
    registry = app_state.get("registry")

    if pipeline is None:
        return {
            "status": "not_ready",
            "reason": "No production model loaded",
            "action": "Run: python -m backend.ml.train",
        }

    prod_version = registry.get_production_version() if registry else None
    return {
        "status": "ready",
        "production_model": prod_version,
        "model_loaded": True,
    }


@router.get(
    "/metrics",
    summary="Model performance metrics",
    description="Returns evaluation metrics for the currently active production model.",
)
async def metrics(request: Request) -> dict:
    from backend.api.main import app_state
    registry = app_state.get("registry")

    if not registry:
        return {"error": "Registry not available"}

    prod_version = registry.get_production_version()
    if not prod_version:
        return {"error": "No production model. Run training first."}

    record = registry.get_version(prod_version)
    eval_metrics = record.get("metrics", {})

    # Python runtime info (useful for debugging environment issues)
    runtime = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    return {
        "production_version": prod_version,
        "model_key": record["model_key"],
        "promoted_at": record.get("promoted_at"),
        "performance": {
            "accuracy":        eval_metrics.get("accuracy"),
            "macro_f1":        eval_metrics.get("macro_f1"),
            "weighted_f1":     eval_metrics.get("weighted_f1"),
            "macro_precision": eval_metrics.get("macro_precision"),
            "macro_recall":    eval_metrics.get("macro_recall"),
            "top2_accuracy":   eval_metrics.get("top2_accuracy"),
        },
        "per_class": eval_metrics.get("per_class", {}),
        "confusion_matrix": eval_metrics.get("confusion_matrix_normalized", []),
        "classes": eval_metrics.get("classes", []),
        "training_info": {
            "n_train":            record["training_info"].get("n_train"),
            "n_test":             record["training_info"].get("n_test"),
            "train_time_seconds": record["training_info"].get("train_time_seconds"),
            "best_cv_accuracy":   record["training_info"].get("best_cv_accuracy"),
            "best_params":        record["training_info"].get("best_params", {}),
        },
        "runtime": runtime,
    }
