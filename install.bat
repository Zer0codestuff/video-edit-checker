@echo off
cd /d "%~dp0"
where py >nul 2>nul && (py -3 install.py) || (python install.py)
pause
