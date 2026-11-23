"""
soundml/backend/utils/config.py
Centralised configuration (no pydantic-settings dependency).
"""
import os
from pathlib import Path
from dataclasses import dataclass, field

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@dataclass
class Settings:
    APP_NAME: str = "SoundML"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: list = field(default_factory=lambda: ["http://localhost:3000","http://localhost:5173"])
    DATA_DIR: Path = BASE_DIR / "backend" / "data"
    MODELS_DIR: Path = BASE_DIR / "backend" / "ml" / "models"
    DATASET_PATH: Path = BASE_DIR / "backend" / "data" / "spotify_dataset.csv"
    REGISTRY_PATH: Path = BASE_DIR / "backend" / "ml" / "models" / "registry.json"
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    DEFAULT_K: int = 5
    MAX_REGISTERED_VERSIONS: int = 10
    AUTO_PROMOTE_THRESHOLD: float = 0.85

settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
