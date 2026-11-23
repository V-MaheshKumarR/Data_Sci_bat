# 🎵 SoundML — Production ML System
## Spotify Audio Features Multi-Class Genre Classifier

A complete, production-grade ML system for music genre classification using Spotify audio features.

---

## 🏗️ Architecture

```
soundml/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app entrypoint
│   │   ├── routes/
│   │   │   ├── predict.py       # POST /predict endpoint
│   │   │   ├── models.py        # Model versioning endpoints
│   │   │   ├── dataset.py       # Dataset CRUD endpoints
│   │   │   └── health.py        # Health & metrics
│   │   └── middleware.py        # CORS, logging, rate limiting
│   ├── ml/
│   │   ├── pipeline.py          # Full sklearn Pipeline
│   │   ├── train.py             # Model training script
│   │   ├── evaluate.py          # Evaluation & metrics
│   │   ├── registry.py          # Model versioning & registry
│   │   └── models/              # Serialized model artifacts
│   ├── data/
│   │   ├── dataset.py           # Dataset generation & loading
│   │   ├── preprocessing.py     # Feature engineering
│   │   └── spotify_dataset.csv  # Raw training data
│   └── utils/
│       ├── config.py            # Settings & env vars
│       └── logger.py            # Structured logging
├── frontend/
│   └── src/
│       └── App.jsx              # React dashboard
├── scripts/
│   ├── train_and_register.sh   # CI/CD model training script
│   └── generate_dataset.py     # Dataset generation utility
├── docker-compose.yml
├── Dockerfile.backend
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset & train model
python scripts/generate_dataset.py
python backend/ml/train.py

# 3. Start API server
uvicorn backend.api.main:app --reload --port 8000

# 4. API docs
open http://localhost:8000/docs
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | Classify a track |
| GET | `/api/v1/models` | List all model versions |
| POST | `/api/v1/models/promote/{version}` | Promote model to production |
| GET | `/api/v1/models/active` | Get current production model |
| GET | `/api/v1/dataset/stats` | Dataset statistics |
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/metrics` | Model performance metrics |

---

## 🧠 ML Pipeline

1. **Data Collection** → Spotify-style audio features (60+ tracks, 6 genres)
2. **Preprocessing** → StandardScaler + outlier detection
3. **Feature Engineering** → Composite features (energy_dance_ratio, etc.)
4. **Model Training** → RandomForest + KNN + SVM with GridSearchCV
5. **Evaluation** → Accuracy, F1, Confusion Matrix, ROC-AUC
6. **Model Registry** → JSON-based versioning with metadata
7. **Serving** → FastAPI with model hot-reloading

---

## 🎓 Educational Value

This system demonstrates:
- Real ML pipeline with sklearn Pipelines
- Model versioning and A/B testing concepts
- REST API design for ML systems
- Feature preprocessing best practices
- Cross-validation and hyperparameter tuning
- Production monitoring patterns
