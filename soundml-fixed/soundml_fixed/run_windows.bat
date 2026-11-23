@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  SoundML — Windows Quick Start
REM  Step 1: Train  (builds fresh models with YOUR sklearn version)
REM  Step 2: Promote best model to production
REM  Step 3: Start server
REM ─────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"

echo.
echo ================================================
echo   Step 1/3 — Training models
echo ================================================
python backend\ml\train.py --no-tune
if %errorlevel% neq 0 (
    echo [ERROR] Training failed. See error above.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Step 2/3 — Promoting best model to production
echo ================================================
python backend\ml\promote_best.py
if %errorlevel% neq 0 (
    echo [WARN] Promotion script failed — server will start without a production model.
)

echo.
echo ================================================
echo   Step 3/3 — Starting API server
echo   Open: http://localhost:8000/docs
echo   Ctrl+C to stop
echo ================================================
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

pause
