@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found.
  echo Install dependencies first or recreate the .venv folder.
  pause
  exit /b 1
)

set PORT=8081
"%~dp0.venv\Scripts\python.exe" "%~dp0website.py" > "%~dp0website.log" 2>&1
