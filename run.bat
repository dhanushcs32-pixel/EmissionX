@echo off
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_industrial.py %*
) else (
    uv run run_industrial.py %*
)

