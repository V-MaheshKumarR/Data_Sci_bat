"""
backend/ml/promote_best.py
───────────────────────────
Reads the registry, finds the highest-accuracy staging (or any) model,
and promotes it to production.

Run after training:
    python backend/ml/promote_best.py
"""

# ── Path bootstrap ────────────────────────────────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ──────────────────────────────────────────────────────────────────────────

from backend.ml.registry import ModelRegistry


def main():
    registry = ModelRegistry()
    data = registry._load()
    versions = data.get("versions", {})

    if not versions:
        print("[ERROR] No models in registry. Run train.bat first.")
        sys.exit(1)

    # Already have a production model?
    current_prod = data.get("production_version")
    if current_prod and current_prod in versions:
        print(f"[INFO] Current production model: {current_prod}")

    # Find the highest-accuracy model among all registered versions
    best_version = None
    best_acc = -1.0
    for version, meta in versions.items():
        acc = meta.get("metrics", {}).get("accuracy", 0.0)
        if acc > best_acc:
            best_acc = acc
            best_version = version

    if best_version is None:
        print("[ERROR] Could not determine best model.")
        sys.exit(1)

    if best_version == current_prod:
        print(f"[OK] Best model '{best_version}' (acc={best_acc:.4f}) is already in production.")
        return

    print(f"[INFO] Promoting '{best_version}' (acc={best_acc:.4f}) to production...")
    registry.promote(best_version)
    print(f"[OK] '{best_version}' is now the production model.")


if __name__ == "__main__":
    main()
