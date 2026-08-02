@echo off
setlocal

chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "HF_HOME=%ROOT%models\backend"
set "NO_PAUSE=0"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

if not exist "%ROOT%backend\.venv\Scripts\python.exe" (
  echo [ERROR] MeloMate backend Python environment was not found.
  echo Run setup-windows.bat or setup-windows-gpu.bat first.
  goto :fail
)

if not exist "%HF_HOME%" mkdir "%HF_HOME%"

"%ROOT%backend\.venv\Scripts\python.exe" -c "import huggingface_hub, omnivoice"
if errorlevel 1 (
  echo [ERROR] Voice cloning dependencies are not installed.
  echo Run setup-windows.bat and choose voice cloning first.
  goto :fail
)

echo Checking the local pinned OmniVoice and Whisper model cache...
"%ROOT%backend\.venv\Scripts\python.exe" "%ROOT%backend\download_voice_clone_models.py" --local-only
if not errorlevel 1 goto :ready

echo.
echo Local model cache is missing or incomplete.
echo You can download all three MeloMate-Model-Cache.7z parts from:
echo https://github.com/misheng521/MeloMate/releases/tag/v1.0.0-models
echo Extract .7z.001 into: %HF_HOME%
echo Then run this script again; no Hugging Face download will be needed.
echo.
echo Downloading the fixed, verified model revisions from Hugging Face now...
echo This model is large. Keep this window open until it finishes.
"%ROOT%backend\.venv\Scripts\python.exe" "%ROOT%backend\download_voice_clone_models.py"
if errorlevel 1 (
  echo.
  echo [ERROR] OmniVoice model download failed.
  echo Check your network, proxy, or Hugging Face access, then run this file again.
  goto :fail
)

:ready
echo.
echo OmniVoice and reference-transcription models are locally verified and ready.
if "%NO_PAUSE%"=="0" pause
exit /b 0

:fail
if "%NO_PAUSE%"=="0" pause
exit /b 1
