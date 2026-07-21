@echo off
setlocal

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "ROOT=%~dp0"
cd /d "%ROOT%"

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js was not found in PATH.
  echo Please install Node.js, then run this file again.
  pause
  exit /b 1
)

if not exist "%ROOT%dist\index.html" (
  echo [ERROR] dist\index.html was not found.
  echo Run npm install and npm run build first.
  pause
  exit /b 1
)

if not exist "%ROOT%backend\.venv\Scripts\python.exe" (
  echo [ERROR] MeloMate backend Python environment was not found.
  echo Expected: %ROOT%backend\.venv\Scripts\python.exe
  echo Run setup-windows.bat first.
  pause
  exit /b 1
)

if not exist "%ROOT%models\backend\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17\model.int8.onnx" (
  echo [WARN] Default local ASR model was not found.
  echo [WARN] Voice recognition may fail unless backend\conf.yaml is changed to another ASR provider.
)

"%ROOT%backend\.venv\Scripts\python.exe" -c "import importlib.util as u; missing=[m for m in ['torchaudio','transformers'] if not u.find_spec(m)]; raise SystemExit(1 if missing else 0)" >nul 2>nul
if errorlevel 1 (
  echo [WARN] OmniVoice voice cloning dependencies are not fully installed.
  echo [WARN] Normal MeloMate audio still works. To enable cloning, install:
  echo [WARN] %ROOT%backend\omnivoice-requirements.txt
)

echo Starting MeloMate...
start "MeloMate Backend" /min "%ROOT%backend\.venv\Scripts\python.exe" "%ROOT%backend\mini_backend.py"
start "" /min cmd /c "timeout /t 1 /nobreak >nul & start "" http://127.0.0.1:5178/"
node server.mjs
