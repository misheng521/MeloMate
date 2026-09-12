@echo off
setlocal
cd /d "%~dp0"
if not exist "backend\.venv\Scripts\python.exe" (
  echo MeloMate's Python environment was not found. Run the main setup first.
  exit /b 1
)
echo Updating browser tools for an older MeloMate installation.
"backend\.venv\Scripts\python.exe" -m pip install -r "backend\pc-tools-requirements.txt"
if errorlevel 1 exit /b 1
"backend\.venv\Scripts\python.exe" "backend\browser_environment.py"
if errorlevel 1 (
  "backend\.venv\Scripts\python.exe" -m playwright install chromium
  if errorlevel 1 exit /b 1
  "backend\.venv\Scripts\python.exe" "backend\browser_environment.py"
  if errorlevel 1 exit /b 1
)
echo Browser tools are prepared. Existing Edge or Chrome was reused when available.
echo Restart MeloMate to load the component.
endlocal
