@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  SoundML — Start API Server (Windows)
REM  Requires training to have been run first (train.bat)
REM ─────────────────────────────────────────────────────────────────────────
cd /d "%~dp0"

echo.
echo ================================================
echo   SoundML API Server
echo   http://localhost:8000
echo   http://localhost:8000/docs   (Swagger UI)
echo   Ctrl+C to stop
echo ================================================
echo.

pip install fastapi uvicorn pydantic 2>nul
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

pause
