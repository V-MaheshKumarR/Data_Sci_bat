"""
soundml/backend/api/main.py
────────────────────────────
FastAPI application entrypoint for SoundML.

Registers:
  - CORS middleware
  - Request logging middleware
  - Rate limiting middleware
  - All API routers
  - Lifespan events (startup model loading)
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from backend.api.routes import predict, models, dataset, health
from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger("soundml.api")

# ── Global model cache ─────────────────────────────────────────────────────
# Loaded once at startup, shared across all requests.
# In production, use a proper cache (Redis) or model server (Triton).
app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle events.

    On startup: load production model into memory.
    On shutdown: clean up resources.
    """
    log.info("SoundML API starting up...")
    try:
        from backend.ml.registry import ModelRegistry
        registry = ModelRegistry()
        pipeline, label_encoder = registry.load_production()
        app_state["pipeline"] = pipeline
        app_state["label_encoder"] = label_encoder
        app_state["registry"] = registry
        prod_version = registry.get_production_version()
        log.info(f"Production model loaded: {prod_version}")
    except RuntimeError as e:
        log.warning(f"No production model found: {e}. Run training first.")
        app_state["pipeline"] = None
        app_state["label_encoder"] = None
        app_state["registry"] = None

    yield  # ← API is now serving requests

    log.info("SoundML API shutting down...")
    app_state.clear()


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## 🎵 SoundML — Spotify Audio Features Genre Classifier API

A production-grade ML API for classifying music genres from Spotify audio features.

### Features
- Multi-class genre classification (Pop, Hip-Hop, Electronic, Rock, R&B, Country)
- K-Nearest Neighbors, Random Forest, and SVM models
- Full model versioning and promotion workflow
- Comprehensive evaluation metrics
- Real-time predictions with confidence scores

### Quick Start
```python
import httpx
response = httpx.post("http://localhost:8000/api/v1/predict", json={
    "danceability": 0.72,
    "energy": 0.85,
    "valence": 0.64,
    "tempo": 128,
    "acousticness": 0.02,
    "speechiness": 0.04
})
print(response.json())
```
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS + ["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    log.info(
        "HTTP Request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# ── Global exception handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "path": str(request.url.path),
        },
    )


# ── Routers ────────────────────────────────────────────────────────────────
PREFIX = settings.API_PREFIX

app.include_router(health.router,   prefix=PREFIX, tags=["Health"])
app.include_router(predict.router,  prefix=PREFIX, tags=["Predictions"])
app.include_router(models.router,   prefix=PREFIX, tags=["Model Registry"])
app.include_router(dataset.router,  prefix=PREFIX, tags=["Dataset"])


# ── Root redirect ──────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs":    "/docs",
        "health":  f"{PREFIX}/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
