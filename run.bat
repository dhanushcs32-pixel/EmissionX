@echo off
if exist "C:\Users\DHANUSH\AppData\Local\Python\bin\python3.11.exe" (
    "C:\Users\DHANUSH\AppData\Local\Python\bin\python3.11.exe" run_industrial.py %*
) else (
    py -3.11 run_industrial.py %*
)

