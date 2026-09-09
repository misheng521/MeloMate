@echo off
setlocal
cd /d "%~dp0"
if not exist "backend\.venv\Scripts\python.exe" (
  echo MeloMate's Python environment was not found. Run the main setup first.
  exit /b 1
)
echo Installing the optional PC browser component into MeloMate's environment.
"backend\.venv\Scripts\python.exe" -m pip install -r "backend\pc-tools-requirements.txt"
if errorlevel 1 exit /b 1
echo The browser tools use installed Microsoft Edge first.
echo No browser or voice model has been downloaded by this script.
echo Restart MeloMate to load the component.
endlocal
