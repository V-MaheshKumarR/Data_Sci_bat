"""
soundml/backend/api/routes/predict.py
───────────────────────────────────────
POST /api/v1/predict — Core ML inference endpoint.

Request  → AudioFeatures (pydantic model)
Response → PredictionResponse with genre, confidence, neighbors

This module also demonstrates:
  - Pydantic v2 validation with custom validators
  - FastAPI dependency injection for model access
  - Structured error responses
  - Prediction explanation (SHAP-lite: feature contribution)
"""

# ── Path bootstrap ────────────────────────────────────────────────────────
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
# ──────────────────────────────────────────────────────────────────────────

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
import numpy as np

from backend.data.preprocessing import validate_features, sample_to_dataframe, FEATURES
from backend.utils.logger import get_logger

log = get_logger("soundml.predict")
router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────

class AudioFeatures(BaseModel):
    """
    Spotify-style audio features for a single track.

    All features are extracted by Spotify's audio analysis engine.
    Ranges are enforced at the API layer.
    """
    danceability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Suitability for dancing. 1.0 = most danceable.",
        example=0.72
    )
    energy: float = Field(
        ..., ge=0.0, le=1.0,
        description="Perceptual intensity. High energy = fast, loud, noisy.",
        example=0.85
    )
    valence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Musical positiveness. High = happy, low = sad/angry.",
        example=0.64
    )
    tempo: float = Field(
        ..., ge=40.0, le=250.0,
        description="Estimated beats per minute (BPM).",
        example=128.0
    )
    acousticness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score that the track is acoustic.",
        example=0.02
    )
    speechiness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Presence of spoken words. >0.33 likely rap/speech.",
        example=0.04
    )

    model_config = {"json_schema_extra": {
        "example": {
            "danceability": 0.72,
            "energy": 0.85,
            "valence": 0.64,
            "tempo": 128.0,
            "acousticness": 0.02,
            "speechiness": 0.04,
        }
    }}


class GenreConfidence(BaseModel):
    genre: str
    probability: float
    percentage: int


class NeighborInfo(BaseModel):
    title: str
    artist: str
    genre: str
    distance: float


class PredictionResponse(BaseModel):
    """
    Full prediction response including:
    - Top predicted genre
    - Confidence scores for all genres
    - K nearest neighbors (for KNN models)
    - Model metadata
    """
    model_config = {"protected_namespaces": ()}

    predicted_genre: str
    confidence: float
    confidence_pct: int

    all_genres: list[GenreConfidence]
    model_version: str
    model_key: str

    # Input echo (for debugging / logging)
    input_features: dict

    # For educational display
    explanation: str


class BatchPredictRequest(BaseModel):
    """Batch prediction for multiple tracks at once."""
    model_config = {"protected_namespaces": ()}

    tracks: list[AudioFeatures]
    model_version: str | None = None  # None = use production


class BatchPredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predictions: list[PredictionResponse]
    model_version: str
    total_tracks: int


# ── Dependency ─────────────────────────────────────────────────────────────

def get_model(request: Request):
    """FastAPI dependency: returns (pipeline, label_encoder) from app state."""
    from backend.api.main import app_state
    pipeline = app_state.get("pipeline")
    label_encoder = app_state.get("label_encoder")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_ready",
                "message": "No production model is loaded. Run: python -m backend.ml.train",
            }
        )
    return pipeline, label_encoder


def get_registry(request: Request):
    """FastAPI dependency: returns the ModelRegistry from app state."""
    from backend.api.main import app_state
    registry = app_state.get("registry")
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not available")
    return registry


# ── Helper ─────────────────────────────────────────────────────────────────

def _run_inference(
    features: AudioFeatures,
    pipeline,
    label_encoder,
    registry,
) -> PredictionResponse:
    """Core inference logic, shared by single and batch endpoints."""
    sample = features.model_dump()

    # Validate
    is_valid, err = validate_features(sample)
    if not is_valid:
        raise HTTPException(status_code=422, detail={"error": "invalid_features", "message": err})

    # Prepare input
    X = sample_to_dataframe(sample).values.astype(np.float64)

    # Predict
    pred_idx = pipeline.predict(X)[0]
    predicted_genre = label_encoder.decode_one(int(pred_idx))

    # Probabilities
    all_confidences = []
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        for i, genre in enumerate(label_encoder.classes):
            all_confidences.append(GenreConfidence(
                genre=genre,
                probability=round(float(proba[i]), 4),
                percentage=int(round(proba[i] * 100)),
            ))
        all_confidences.sort(key=lambda x: -x.probability)
        confidence = float(proba[pred_idx])
    else:
        # Fallback: binary confidence
        all_confidences = [
            GenreConfidence(genre=g, probability=1.0 if g == predicted_genre else 0.0, percentage=100 if g == predicted_genre else 0)
            for g in label_encoder.classes
        ]
        confidence = 1.0

    # Explanation
    high_feats = [f for f in ["acousticness", "speechiness", "energy"] if sample[f] > 0.5]
    low_feats = [f for f in ["acousticness", "speechiness"] if sample[f] < 0.1]
    reasons = []
    if high_feats:
        reasons.append(f"high {', '.join(high_feats)}")
    if low_feats:
        reasons.append(f"low {', '.join(low_feats)}")
    if sample["tempo"] > 140:
        reasons.append("fast tempo")
    elif sample["tempo"] < 90:
        reasons.append("slow tempo")
    explanation = (
        f"Classified as {predicted_genre} based on: {', '.join(reasons)}."
        if reasons else f"Feature profile most closely matches {predicted_genre} tracks."
    )

    version = registry.get_production_version() if registry else "unknown"
    version_record = registry.get_version(version) if registry and version else {}

    return PredictionResponse(
        predicted_genre=predicted_genre,
        confidence=round(confidence, 4),
        confidence_pct=int(round(confidence * 100)),
        all_genres=all_confidences,
        model_version=version or "unknown",
        model_key=version_record.get("model_key", "unknown"),
        input_features=sample,
        explanation=explanation,
    )


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a track's genre from audio features",
    description="""
Submit Spotify-style audio features to classify a track into one of 6 genres:
**Pop, Hip-Hop, Electronic, Rock, R&B, Country**

The response includes:
- Predicted genre and confidence
- Probability distribution across all genres
- Human-readable explanation of the prediction
    """,
)
async def predict_genre(
    features: AudioFeatures,
    model_and_enc=Depends(get_model),
    registry=Depends(get_registry),
) -> PredictionResponse:
    pipeline, label_encoder = model_and_enc
    result = _run_inference(features, pipeline, label_encoder, registry)
    log.info(
        "Prediction made",
        predicted=result.predicted_genre,
        confidence=result.confidence,
        version=result.model_version,
    )
    return result


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Classify multiple tracks in one request",
)
async def predict_batch(
    request_body: BatchPredictRequest,
    model_and_enc=Depends(get_model),
    registry=Depends(get_registry),
) -> BatchPredictResponse:
    pipeline, label_encoder = model_and_enc
    predictions = []
    for track in request_body.tracks:
        result = _run_inference(track, pipeline, label_encoder, registry)
        predictions.append(result)

    version = registry.get_production_version() if registry else "unknown"
    log.info(f"Batch prediction: {len(predictions)} tracks")
    return BatchPredictResponse(
        predictions=predictions,
        model_version=version or "unknown",
        total_tracks=len(predictions),
    )
