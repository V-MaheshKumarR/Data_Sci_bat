@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  SoundML — Train + Promote (Windows)
REM  Run this FIRST before starting the server.
REM  This builds models using YOUR local sklearn version (no version errors).
REM ─────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"

echo.
echo ================================================
echo   SoundML — Training models
echo ================================================
python backend\ml\train.py --no-tune
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Training failed.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Promoting best model to production
echo ================================================
python backend\ml\promote_best.py

echo.
echo [OK] Done! Now run:  run_server.bat
pause
