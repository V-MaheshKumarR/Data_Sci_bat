"""
backend/utils/config.py
────────────────────────
Centralised settings.  Bootstraps sys.path FIRST so every other
module that imports config will automatically have the project root
on the path — no matter how the script was invoked.
"""

# ── Path bootstrap (must be first) ────────────────────────────────────────
import sys
from pathlib import Path

# This file lives at:  <project_root>/backend/utils/config.py
# parents[0] = backend/utils
# parents[1] = backend
# parents[2] = <project_root>   ← the folder that contains 'backend/'
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Settings ───────────────────────────────────────────────────────────────
from dataclasses import dataclass, field

BASE_DIR = _PROJECT_ROOT


@dataclass
class Settings:
    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "SoundML"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # ── API ────────────────────────────────────────────────────────────────
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: list = field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"]
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / "backend" / "data"
    MODELS_DIR: Path = BASE_DIR / "backend" / "ml" / "models"
    DATASET_PATH: Path = BASE_DIR / "backend" / "data" / "spotify_dataset.csv"
    REGISTRY_PATH: Path = BASE_DIR / "backend" / "ml" / "models" / "registry.json"

    # ── ML ─────────────────────────────────────────────────────────────────
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5
    DEFAULT_K: int = 5

    # ── Model Registry ─────────────────────────────────────────────────────
    MAX_REGISTERED_VERSIONS: int = 10
    AUTO_PROMOTE_THRESHOLD: float = 0.85


settings = Settings()

# Ensure directories exist at import time
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
