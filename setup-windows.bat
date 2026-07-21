@echo off
setlocal

chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js was not found in PATH.
  echo Please install Node.js 20 or newer, then run this script again.
  goto :fail
)

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  echo Please install Python 3.11 and enable "Add python.exe to PATH".
  goto :fail
)

echo Node version:
node --version
echo Python version:
python --version
echo.

echo [1/3] Installing frontend dependencies...
call npm install
if errorlevel 1 (
  echo [ERROR] npm install failed.
  goto :fail
)

echo [2/3] Creating backend virtual environment...
if not exist "%ROOT%backend\.venv\Scripts\python.exe" (
  python -m venv "%ROOT%backend\.venv"
  if errorlevel 1 (
    echo [ERROR] Failed to create backend virtual environment.
    goto :fail
  )
)

echo [3/3] Installing backend dependencies...
"%ROOT%backend\.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.
  goto :fail
)
"%ROOT%backend\.venv\Scripts\python.exe" -m pip install -r "%ROOT%backend\requirements.txt"
if errorlevel 1 (
  echo [ERROR] Failed to install backend dependencies.
  goto :fail
)

echo Verifying backend dependencies...
"%ROOT%backend\.venv\Scripts\python.exe" -c "import importlib.util as u; mods=['fastapi','uvicorn','websockets','loguru','pydantic','yaml','numpy','soundfile','httpx','requests','aiohttp','openai','anthropic','edge_tts','pysbd','langdetect','pydub','sherpa_onnx','onnxruntime','multipart','chardet','jinja2','tqdm','mcp','letta_client','torch','torchaudio','transformers','accelerate','librosa']; missing=[m for m in mods if u.find_spec(m) is None]; print('Missing backend dependencies: '+', '.join(missing)) if missing else print('Backend dependencies verified.'); raise SystemExit(1 if missing else 0)"
if errorlevel 1 (
  echo [ERROR] Backend dependency verification failed.
  goto :fail
)

echo.
echo Setup finished. Configure backend\conf.yaml, then run start.bat.
pause
exit /b 0

:fail
echo.
echo Setup failed. Read the error message above.
pause
exit /b 1
