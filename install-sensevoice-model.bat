@echo off
setlocal

chcp 65001 >nul
set "ROOT=%~dp0"
set "ARCHIVE=%~1"
if not defined ARCHIVE set "ARCHIVE=%ROOT%sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.7z"

if not exist "%ROOT%backend\.venv\Scripts\python.exe" (
  echo [ERROR] MeloMate backend Python environment was not found.
  echo Run setup-windows.bat first.
  exit /b 1
)
if not exist "%ARCHIVE%" (
  echo [ERROR] SenseVoice archive was not found: %ARCHIVE%
  echo Download the fixed archive from:
  echo https://github.com/misheng521/MeloMate/releases/tag/models-v0.1.0
  exit /b 1
)

"%ROOT%backend\.venv\Scripts\python.exe" "%ROOT%backend\install_sensevoice_release.py" "%ARCHIVE%"
if errorlevel 1 exit /b 1
echo SenseVoice is ready. The verified local model will be used without downloading.
exit /b 0
