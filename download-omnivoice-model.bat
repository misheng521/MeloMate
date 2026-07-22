@echo off
setlocal

chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist "%ROOT%backend\.venv\Scripts\python.exe" (
  echo [ERROR] MeloMate backend Python environment was not found.
  echo Run setup-windows.bat or setup-windows-gpu.bat first.
  pause
  exit /b 1
)

echo Downloading OmniVoice model k2-fsa/OmniVoice...
echo This model is large. Keep this window open until it finishes.
echo.

"%ROOT%backend\.venv\Scripts\python.exe" -m pip install huggingface_hub
if errorlevel 1 (
  echo [ERROR] Failed to install huggingface_hub.
  pause
  exit /b 1
)

"%ROOT%backend\.venv\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; path=snapshot_download(repo_id='k2-fsa/OmniVoice'); print('OmniVoice model downloaded to:', path)"
if errorlevel 1 (
  echo.
  echo [ERROR] OmniVoice model download failed.
  echo Check your network, proxy, or Hugging Face access, then run this file again.
  pause
  exit /b 1
)

echo.
echo OmniVoice model download finished.
pause
exit /b 0
