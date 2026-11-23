#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
#  SoundML — Mac / Linux Quick Start
#  chmod +x run.sh && ./run.sh
# ─────────────────────────────────────────────────────────────────────────
set -e

# Always run from the project root (where this script lives)
cd "$(dirname "$0")"

echo ""
echo "================================================"
echo "  SoundML — Installing dependencies"
echo "================================================"
pip install scikit-learn numpy pandas joblib fastapi uvicorn pydantic 2>/dev/null || true

echo ""
echo "================================================"
echo "  Step 1/2 — Training models"
echo "================================================"
python backend/ml/train.py --no-tune

echo ""
echo "================================================"
echo "  Step 2/2 — Starting API server"
echo "  Open: http://localhost:8000/docs"
echo "  Ctrl+C to stop"
echo "================================================"
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
