"""
soundml/backend/api/routes/dataset.py + health.py (combined)
──────────────────────────────────────────────────────────────
Dataset management and health/metrics endpoints.
"""

# ═══════════════════════════════════════════════════════════════════════════
# dataset.py
# ═══════════════════════════════════════════════════════════════════════════

from fastapi import APIRouter, HTTPException, Request
import pandas as pd
import numpy as np

from backend.data.dataset import build_full_dataset, get_dataset_stats, GENRES, FEATURES
from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger("soundml.dataset_api")
router = APIRouter()


@router.get(
    "/dataset/stats",
    summary="Get dataset statistics",
)
async def dataset_stats() -> dict:
    """Returns descriptive statistics for the training dataset."""
    try:
        if settings.DATASET_PATH.exists():
            df = pd.read_csv(settings.DATASET_PATH)
        else:
            df = build_full_dataset(save_path=settings.DATASET_PATH)
        return get_dataset_stats(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/dataset/tracks",
    summary="List tracks in the dataset",
)
async def list_tracks(
    genre: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Returns paginated tracks from the dataset, with optional genre filter."""
    if not settings.DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Run training first.")

    df = pd.read_csv(settings.DATASET_PATH)
    if genre:
        if genre not in GENRES:
            raise HTTPException(status_code=400, detail=f"Invalid genre. Choose from: {GENRES}")
        df = df[df["genre"] == genre]

    total = len(df)
    page = df.iloc[offset: offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "genre_filter": genre,
        "tracks": page.replace({np.nan: None}).to_dict(orient="records"),
    }


@router.post(
    "/dataset/regenerate",
    summary="Regenerate the synthetic training dataset",
)
async def regenerate_dataset(
    n_synthetic_per_genre: int = 50,
    seed: int = 42,
) -> dict:
    """Regenerates the training dataset with new synthetic samples."""
    df = build_full_dataset(
        n_synthetic_per_genre=n_synthetic_per_genre,
        save_path=settings.DATASET_PATH,
        seed=seed,
    )
    return {
        "message": "Dataset regenerated successfully",
        "total_rows": len(df),
        "genre_counts": df["genre"].value_counts().to_dict(),
        "saved_to": str(settings.DATASET_PATH),
    }
