"""
soundml/backend/api/routes/models.py
───────────────────────────────────────
Model registry management endpoints.

Endpoints:
  GET  /models           → List all registered versions
  GET  /models/active    → Get production model info
  GET  /models/{version} → Get specific version details
  POST /models/promote/{version} → Promote to production
  POST /models/demote/{version}  → Demote (rollback)
  DELETE /models/{version}       → Delete a staging/archived version
  POST /models/train             → Trigger retraining (background task)
"""

# ── Path bootstrap ────────────────────────────────────────────────────────
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
# ──────────────────────────────────────────────────────────────────────────

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Optional

from backend.utils.logger import get_logger

log = get_logger("soundml.models_api")
router = APIRouter()


# ── Response schemas ───────────────────────────────────────────────────────

class VersionSummary(BaseModel):
    version: str
    model_key: str
    status: str          # staging | production | archived
    created_at: str
    promoted_at: Optional[str]
    accuracy: Optional[float]
    macro_f1: Optional[float]


class VersionDetail(BaseModel):
    version: str
    model_key: str
    status: str
    created_at: str
    promoted_at: Optional[str]
    archived_at: Optional[str]
    metrics: dict
    training_info: dict
    dataset_info: dict


class RegistrySummary(BaseModel):
    production_version: Optional[str]
    total_versions: int
    versions: list[VersionSummary]


class TrainRequest(BaseModel):
    models: Optional[list[str]] = None   # None = train all
    tune: bool = False                    # Quick train by default via API
    synthetic_per_genre: int = 50
    auto_promote: bool = True


class TrainResponse(BaseModel):
    message: str
    job_id: str


# ── Dependency ─────────────────────────────────────────────────────────────

def get_registry(request: Request):
    from backend.api.main import app_state
    registry = app_state.get("registry")
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not available")
    return registry


def _to_version_summary(record: dict) -> VersionSummary:
    metrics = record.get("metrics", {})
    return VersionSummary(
        version=record["version"],
        model_key=record["model_key"],
        status=record["status"],
        created_at=record["created_at"],
        promoted_at=record.get("promoted_at"),
        accuracy=metrics.get("accuracy"),
        macro_f1=metrics.get("macro_f1"),
    )


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get(
    "/models",
    response_model=RegistrySummary,
    summary="List all registered model versions",
)
async def list_models(
    model_key: Optional[str] = None,
    status: Optional[str] = None,
    registry=None,
    request: Request = None,
) -> RegistrySummary:
    registry = get_registry(request)
    versions = registry.list_versions(model_key=model_key)
    if status:
        versions = [v for v in versions if v["status"] == status]

    summary = registry.summary()
    return RegistrySummary(
        production_version=summary["production_version"],
        total_versions=summary["total_versions"],
        versions=[_to_version_summary(v) for v in versions],
    )


@router.get(
    "/models/active",
    summary="Get the current production model details",
)
async def get_active_model(request: Request) -> dict:
    registry = get_registry(request)
    prod_version = registry.get_production_version()
    if not prod_version:
        raise HTTPException(
            status_code=404,
            detail="No production model. Run training: python -m backend.ml.train"
        )
    record = registry.get_version(prod_version)
    return {
        "version": record["version"],
        "model_key": record["model_key"],
        "status": record["status"],
        "promoted_at": record.get("promoted_at"),
        "metrics": record["metrics"],
        "training_info": {
            k: v for k, v in record["training_info"].items()
            if k not in ["cv_results_mean", "cv_results_std"]  # Filter verbose fields
        },
    }


@router.get(
    "/models/{version}",
    response_model=VersionDetail,
    summary="Get details for a specific model version",
)
async def get_model_version(version: str, request: Request) -> VersionDetail:
    registry = get_registry(request)
    try:
        record = registry.get_version(version)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return VersionDetail(
        version=record["version"],
        model_key=record["model_key"],
        status=record["status"],
        created_at=record["created_at"],
        promoted_at=record.get("promoted_at"),
        archived_at=record.get("archived_at"),
        metrics=record["metrics"],
        training_info=record["training_info"],
        dataset_info=record["dataset_info"],
    )


@router.post(
    "/models/promote/{version}",
    summary="Promote a model version to production",
    description="Promotes a staging or archived model to production. The current production model is automatically archived.",
)
async def promote_model(version: str, request: Request) -> dict:
    registry = get_registry(request)
    from backend.api.main import app_state

    try:
        registry.promote(version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Hot-reload: load the newly promoted model into app_state
    try:
        pipeline, label_encoder = registry.load_version(version)
        app_state["pipeline"] = pipeline
        app_state["label_encoder"] = label_encoder
        log.info(f"Hot-reloaded production model: {version}")
    except Exception as e:
        log.warning(f"Could not hot-reload model: {e}")

    return {
        "message": f"Model {version} promoted to production",
        "version": version,
        "status": "production",
    }


@router.post(
    "/models/demote/{version}",
    summary="Demote (rollback) the production model",
)
async def demote_model(version: str, request: Request) -> dict:
    registry = get_registry(request)
    try:
        registry.demote(version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from backend.api.main import app_state
    app_state["pipeline"] = None
    app_state["label_encoder"] = None

    return {
        "message": f"Model {version} demoted. No production model active.",
        "version": version,
        "status": "staging",
    }


@router.delete(
    "/models/{version}",
    summary="Delete a staging or archived model version",
)
async def delete_model(version: str, request: Request) -> dict:
    registry = get_registry(request)
    try:
        registry.delete_version(version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": f"Version {version} deleted", "version": version}


@router.post(
    "/models/train",
    response_model=TrainResponse,
    summary="Trigger model retraining (async background task)",
    description="Starts a background training job. Monitor progress in server logs.",
)
async def trigger_training(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> TrainResponse:
    import uuid
    from backend.api.main import app_state

    job_id = str(uuid.uuid4())[:8]
    registry = get_registry(request)

    def background_train():
        log.info(f"Background training job {job_id} started")
        try:
            from backend.ml.train import train
            result = train(
                model_keys=train_request.models,
                tune=train_request.tune,
                n_synthetic_per_genre=train_request.synthetic_per_genre,
                auto_promote=train_request.auto_promote,
            )
            # Reload production model after training
            if train_request.auto_promote:
                try:
                    pipeline, label_encoder = registry.load_production()
                    app_state["pipeline"] = pipeline
                    app_state["label_encoder"] = label_encoder
                    log.info(f"Job {job_id}: hot-reloaded new production model")
                except Exception as e:
                    log.warning(f"Job {job_id}: hot-reload failed: {e}")
            log.info(f"Background training job {job_id} complete", **result)
        except Exception as e:
            log.error(f"Background training job {job_id} failed: {e}")

    background_tasks.add_task(background_train)
    return TrainResponse(
        message=f"Training job {job_id} started in background. Check server logs.",
        job_id=job_id,
    )
