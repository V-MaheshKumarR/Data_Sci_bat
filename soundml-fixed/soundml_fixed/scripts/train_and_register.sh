#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# soundml/scripts/train_and_register.sh
# CI/CD model training and registration script
#
# Usage:
#   ./scripts/train_and_register.sh                    # Full training with tuning
#   ./scripts/train_and_register.sh --quick            # No hyperparameter tuning
#   ./scripts/train_and_register.sh --model knn        # Single model
#   PROMOTE_THRESHOLD=0.90 ./scripts/train_and_register.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE="$PROJECT_ROOT/logs/train_${TIMESTAMP//:/}.log"
mkdir -p "$PROJECT_ROOT/logs"

# Defaults
TUNE="--tune"
MODELS=""
SYNTHETIC=50
PROMOTE_THRESHOLD="${PROMOTE_THRESHOLD:-0.85}"

# ── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)     TUNE="--no-tune";      shift ;;
    --model)     MODELS="--models $2";  shift 2 ;;
    --synthetic) SYNTHETIC="$2";        shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Header ─────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  SoundML Training Pipeline"
echo "  Started: $TIMESTAMP"
echo "  Tune: $TUNE | Synthetic/genre: $SYNTHETIC"
echo "═══════════════════════════════════════════════════"

cd "$PROJECT_ROOT"

# ── 1. Environment check ───────────────────────────────────────────────────
echo ""
echo "[ 1/5 ] Checking environment..."
python -c "import sklearn, pandas, numpy, fastapi; print('✅ All dependencies installed')"

# ── 2. Generate dataset ────────────────────────────────────────────────────
echo ""
echo "[ 2/5 ] Generating dataset (synthetic=$SYNTHETIC per genre)..."
python -c "
from backend.data.dataset import build_full_dataset
from backend.utils.config import settings
df = build_full_dataset(n_synthetic_per_genre=$SYNTHETIC, save_path=settings.DATASET_PATH)
print(f'✅ Dataset: {len(df)} rows')
print(df[\"genre\"].value_counts().to_string())
" 2>&1 | tee -a "$LOG_FILE"

# ── 3. Train models ────────────────────────────────────────────────────────
echo ""
echo "[ 3/5 ] Training models..."
python -m backend.ml.train \
  $TUNE \
  $MODELS \
  --synthetic "$SYNTHETIC" \
  2>&1 | tee -a "$LOG_FILE"

# ── 4. Validate production model ───────────────────────────────────────────
echo ""
echo "[ 4/5 ] Validating production model..."
python -c "
from backend.ml.registry import ModelRegistry
registry = ModelRegistry()
prod_version = registry.get_production_version()
if not prod_version:
    print('❌ No production model found!')
    exit(1)

record = registry.get_version(prod_version)
acc = record['metrics']['accuracy']
threshold = $PROMOTE_THRESHOLD

print(f'Production model: {prod_version}')
print(f'Accuracy: {acc:.4f} (threshold: {threshold})')

if acc < threshold:
    print(f'❌ Accuracy {acc:.4f} below threshold {threshold}')
    exit(1)

print(f'✅ Model validated successfully')
" 2>&1 | tee -a "$LOG_FILE"

# ── 5. Smoke test the API ──────────────────────────────────────────────────
echo ""
echo "[ 5/5 ] Smoke testing inference..."
python -c "
from backend.ml.registry import ModelRegistry
from backend.data.preprocessing import sample_to_dataframe
import numpy as np

registry = ModelRegistry()
pipeline, label_encoder = registry.load_production()

test_samples = [
    {'name': 'Electronic-like', 'danceability': 0.65, 'energy': 0.92, 'valence': 0.55, 'tempo': 128, 'acousticness': 0.01, 'speechiness': 0.02},
    {'name': 'Country-like',    'danceability': 0.45, 'energy': 0.50, 'valence': 0.70, 'tempo': 90,  'acousticness': 0.55, 'speechiness': 0.04},
    {'name': 'Hip-Hop-like',    'danceability': 0.80, 'energy': 0.65, 'valence': 0.45, 'tempo': 95,  'acousticness': 0.10, 'speechiness': 0.20},
]

for s in test_samples:
    name = s.pop('name')
    X = sample_to_dataframe(s).values.astype(float)
    pred_idx = pipeline.predict(X)[0]
    pred = label_encoder.decode_one(int(pred_idx))
    print(f'  {name:<20} → {pred}')

print('✅ Smoke test passed')
" 2>&1 | tee -a "$LOG_FILE"

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ Training pipeline complete!"
echo "  Log: $LOG_FILE"
echo "  Start API: uvicorn backend.api.main:app --reload"
echo "═══════════════════════════════════════════════════"
