@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Esegui prima l'installazione: doppio clic su install.bat
    pause
    exit /b 1
)
.venv\Scripts\python.exe app.py
pause
