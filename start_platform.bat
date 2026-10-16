@echo off
REM Starts the EmissionX backend (FastAPI, port 8000) and website (Vite, port 3000) in two windows,
REM then opens the site. The website reaches the backend through the Vite /api proxy (vite.config.ts).
REM For the GenAI chatbot, set MIMO_API_KEY, MIMO_BASE_URL and MIMO_MODEL in this window first.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=py"
)

if not exist "node_modules" (
    echo Installing website dependencies...
    call npm.cmd install
)

start "EmissionX backend" cmd /k %PY% -m uvicorn main:app --reload --port 8000
start "EmissionX website" cmd /k npm.cmd run dev
timeout /t 5 /nobreak >nul
start "" http://localhost:3000
