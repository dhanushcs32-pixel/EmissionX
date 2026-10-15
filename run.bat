@echo off
set PYTHON_EXE=C:\Users\DHANUSH\AppData\Local\Python\bin\python3.11.exe
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" run_industrial.py %*
) else (
    python run_industrial.py %*
)
