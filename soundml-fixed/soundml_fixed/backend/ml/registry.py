"""
soundml/backend/ml/registry.py
────────────────────────────────
Model versioning and registry system.

Each registered model gets:
  • A unique version string:  {model_key}-v{N}  e.g. "random_forest-v3"
  • A serialised artifact:    models/{version}.joblib
  • A JSON metadata record:   models/registry.json

States:
  staging    → newly registered, not yet serving traffic
  production → the single active model serving inference requests
  archived   → superseded, kept for rollback

This mimics real MLOps patterns (MLflow, Vertex AI Model Registry).
"""

# ── Path bootstrap ────────────────────────────────────────────────────────
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
# ──────────────────────────────────────────────────────────────────────────

import json
import joblib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger("soundml.registry")


class ModelRegistry:
    """
    File-based model registry.

    Production systems use databases or cloud stores (S3/GCS).
    This JSON-based implementation is educational and requires
    no external dependencies.

    Registry file schema (registry.json):
    {
      "versions": {
        "random_forest-v1": {
          "version":      "random_forest-v1",
          "model_key":    "random_forest",
          "status":       "production",    // staging | production | archived
          "created_at":   "2024-01-01T...",
          "promoted_at":  "2024-01-01T...",
          "metrics":      {...},
          "training_info": {...},
          "dataset_info": {...},
          "artifact_path": "random_forest-v1.joblib"
        }
      },
      "production_version": "random_forest-v1"
    }
    """

    def __init__(self):
        self.registry_path = settings.REGISTRY_PATH
        self.models_dir = settings.MODELS_DIR
        self._registry = self._load()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"versions": {}, "production_version": None}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def _next_version_number(self, model_key: str) -> int:
        existing = [
            v for v in self._registry["versions"]
            if v.startswith(f"{model_key}-v")
        ]
        if not existing:
            return 1
        nums = [int(v.split("-v")[-1]) for v in existing]
        return max(nums) + 1

    # ── Public API ─────────────────────────────────────────────────────────

    def register(
        self,
        model_key: str,
        pipeline: Any,
        label_encoder: Any,
        metrics: dict,
        training_info: dict,
        dataset_info: dict,
    ) -> str:
        """
        Serialize and register a trained pipeline.

        Args:
            model_key:     One of 'knn', 'random_forest', 'svm'
            pipeline:      Fitted sklearn Pipeline
            label_encoder: Fitted GenreLabelEncoder
            metrics:       Evaluation metrics dict
            training_info: Hyperparams, timing, etc.
            dataset_info:  Dataset provenance info

        Returns:
            version string (e.g. 'random_forest-v1')
        """
        n = self._next_version_number(model_key)
        version = f"{model_key}-v{n}"
        artifact_filename = f"{version}.joblib"
        artifact_path = self.models_dir / artifact_filename

        # Serialize the pipeline + label encoder as a bundle
        bundle = {
            "pipeline":      pipeline,
            "label_encoder": label_encoder,
            "version":       version,
            "model_key":     model_key,
        }
        joblib.dump(bundle, artifact_path, compress=3)
        log.info(f"Saved artifact: {artifact_path}")

        # Build registry record
        record = {
            "version":       version,
            "model_key":     model_key,
            "status":        "staging",
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "promoted_at":   None,
            "archived_at":   None,
            "metrics":       metrics,
            "training_info": training_info,
            "dataset_info":  dataset_info,
            "artifact_path": artifact_filename,
        }
        self._registry["versions"][version] = record
        self._save()

        log.info(f"Model registered — version={version}, status=staging")

        # Prune old versions if over limit
        self._prune_old_versions(model_key)

        return version

    def promote(self, version: str) -> None:
        """
        Promote a model version to production.

        Automatically archives the previous production model.
        """
        if version not in self._registry["versions"]:
            raise ValueError(f"Version '{version}' not found in registry")

        # Archive current production model
        current_prod = self._registry.get("production_version")
        if current_prod and current_prod != version:
            self._registry["versions"][current_prod]["status"] = "archived"
            self._registry["versions"][current_prod]["archived_at"] = (
                datetime.now(timezone.utc).isoformat()
            )
            log.info(f"Archived previous production model: {current_prod}")

        # Promote new version
        self._registry["versions"][version]["status"] = "production"
        self._registry["versions"][version]["promoted_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        self._registry["production_version"] = version
        self._save()

        log.info(f"Promoted to production: {version}")

    def demote(self, version: str) -> None:
        """Roll back: demote the current production model to staging."""
        if version not in self._registry["versions"]:
            raise ValueError(f"Version '{version}' not found")
        record = self._registry["versions"][version]
        if record["status"] != "production":
            raise ValueError(f"Version '{version}' is not in production (status: {record['status']})")
        record["status"] = "staging"
        self._registry["production_version"] = None
        self._save()
        log.info(f"Demoted: {version}")

    def load_production(self) -> tuple[Any, Any]:
        """
        Load the production model bundle.

        Returns:
            (pipeline, label_encoder)

        Raises:
            RuntimeError if no production model is registered.
        """
        prod_version = self._registry.get("production_version")
        if not prod_version:
            raise RuntimeError(
                "No production model registered. Run training first: "
                "python -m backend.ml.train"
            )
        return self.load_version(prod_version)

    def load_version(self, version: str) -> tuple[Any, Any]:
        """Load a specific model version by its version string."""
        if version not in self._registry["versions"]:
            raise ValueError(f"Version '{version}' not in registry")

        record = self._registry["versions"][version]
        artifact_path = self.models_dir / record["artifact_path"]

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        bundle = joblib.load(artifact_path)
        return bundle["pipeline"], bundle["label_encoder"]

    def get_production_version(self) -> str | None:
        return self._registry.get("production_version")

    def list_versions(self, model_key: str | None = None) -> list[dict]:
        """List all registered versions, optionally filtered by model_key."""
        versions = list(self._registry["versions"].values())
        if model_key:
            versions = [v for v in versions if v["model_key"] == model_key]
        return sorted(versions, key=lambda v: v["created_at"], reverse=True)

    def get_version(self, version: str) -> dict:
        if version not in self._registry["versions"]:
            raise ValueError(f"Version '{version}' not found")
        return self._registry["versions"][version]

    def delete_version(self, version: str) -> None:
        """Delete a version (cannot delete production model)."""
        record = self.get_version(version)
        if record["status"] == "production":
            raise ValueError("Cannot delete the production model. Demote it first.")

        artifact_path = self.models_dir / record["artifact_path"]
        if artifact_path.exists():
            artifact_path.unlink()

        del self._registry["versions"][version]
        self._save()
        log.info(f"Deleted version: {version}")

    def _prune_old_versions(self, model_key: str) -> None:
        """Keep only the N most recent versions per model (excluding production)."""
        all_versions = [
            v for v in self._registry["versions"].values()
            if v["model_key"] == model_key and v["status"] == "archived"
        ]
        all_versions.sort(key=lambda v: v["created_at"])

        max_to_keep = settings.MAX_REGISTERED_VERSIONS
        to_delete = all_versions[:-max_to_keep] if len(all_versions) > max_to_keep else []

        for record in to_delete:
            artifact = self.models_dir / record["artifact_path"]
            if artifact.exists():
                artifact.unlink()
            del self._registry["versions"][record["version"]]
            log.info(f"Pruned old version: {record['version']}")

        if to_delete:
            self._save()

    def summary(self) -> dict:
        """Return a summary of the registry state."""
        versions = self._registry["versions"]
        by_status = {"staging": [], "production": [], "archived": []}
        for v in versions.values():
            by_status[v["status"]].append(v["version"])
        return {
            "production_version": self._registry.get("production_version"),
            "total_versions":     len(versions),
            "by_status":          by_status,
        }
