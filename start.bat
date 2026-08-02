@echo off
setlocal

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "ROOT=%~dp0"
cd /d "%ROOT%"
set "HF_HOME=%ROOT%models\backend"
set "MODELSCOPE_CACHE=%ROOT%models\backend"

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

"%ROOT%backend\.venv\Scripts\python.exe" -c "import importlib.util as u; mods=['fastapi','uvicorn','websockets','loguru','pydantic','yaml','numpy','soundfile','httpx','requests','aiohttp','openai','anthropic','edge_tts','pysbd','langdetect','pydub','sherpa_onnx','onnxruntime','multipart','chardet','jinja2','tqdm','mcp','letta_client','win32crypt']; missing=[m for m in mods if u.find_spec(m) is None]; print(', '.join(missing)); raise SystemExit(1 if missing else 0)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] MeloMate backend dependencies are not installed correctly.
  echo Run setup-windows.bat and wait until it says "Setup finished."
  pause
  exit /b 1
)

"%ROOT%backend\.venv\Scripts\python.exe" "%ROOT%backend\check_runtime_dependencies.py"
if errorlevel 1 (
  echo [ERROR] MeloMate backend dependency versions are incompatible.
  echo Run setup-windows.bat again to install the supported versions.
  pause
  exit /b 1
)

if not exist "%ROOT%models\backend\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17\model.int8.onnx" (
  echo [WARN] Default local ASR model was not found.
  echo [WARN] Voice recognition may fail unless backend\conf.yaml is changed to another ASR provider.
)

"%ROOT%backend\.venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, r'%ROOT%backend'); from src.open_llm_vtuber.utils.optional_dependencies import voice_clone_dependencies_available; raise SystemExit(0 if voice_clone_dependencies_available() else 1)" >nul 2>nul
if errorlevel 1 (
  echo [WARN] OmniVoice voice cloning dependencies are not fully installed.
  echo [WARN] Normal MeloMate audio still works.
  echo [WARN] To enable cloning, rerun setup-windows.bat and choose Y.
)

echo Starting MeloMate...
echo Backend Python: %ROOT%backend\.venv\Scripts\python.exe
"%ROOT%backend\.venv\Scripts\python.exe" "%ROOT%backend\launch_melomate.py"
set "START_RESULT=%ERRORLEVEL%"
if not "%START_RESULT%"=="0" (
  echo.
  echo MeloMate did not start. No existing port owner was terminated.
  if not "%MELOMATE_NO_PAUSE%"=="1" pause
)
exit /b %START_RESULT%
