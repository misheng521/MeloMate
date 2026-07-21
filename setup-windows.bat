@echo off
setlocal

chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js was not found in PATH.
  exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  exit /b 1
)

echo [1/3] Installing frontend dependencies...
npm install
if errorlevel 1 exit /b 1

echo [2/3] Creating backend virtual environment...
if not exist "%ROOT%backend\.venv\Scripts\python.exe" (
  python -m venv "%ROOT%backend\.venv"
  if errorlevel 1 exit /b 1
)

echo [3/3] Installing backend dependencies...
"%ROOT%backend\.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%ROOT%backend\.venv\Scripts\python.exe" -m pip install -r "%ROOT%backend\requirements.txt"
if errorlevel 1 exit /b 1

echo.
echo Setup finished. Configure backend\conf.yaml, then run start.bat.
