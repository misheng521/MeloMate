@echo off
setlocal EnableExtensions

chcp 65001 >nul
set "ROOT=%~dp0"
set "VENV_PYTHON=%ROOT%backend\.venv\Scripts\python.exe"
set "OPTIONAL_REQUESTED=0"
set "OPTIONAL_FAILED=0"
set "VOICE_DEVICE=cpu"
cd /d "%ROOT%"

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js was not found in PATH.
  echo Install Node.js 20 or newer, then run this script again.
  goto :fail
)
node -e "const major=Number(process.versions.node.split('.')[0]); process.exit(major >= 20 ? 0 : 1)"
if errorlevel 1 (
  echo [ERROR] MeloMate requires Node.js 20 or newer.
  node --version
  goto :fail
)

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  echo Install 64-bit Python 3.11 and enable Add python.exe to PATH.
  goto :fail
)
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
  echo [ERROR] MeloMate requires Python 3.11.
  python --version
  goto :fail
)

echo Node version:
node --version
echo Python version:
python --version
echo.

echo [1/7] Installing reproducible frontend dependencies...
call npm ci
if errorlevel 1 (
  echo [ERROR] npm ci failed.
  goto :fail
)

echo [2/7] Creating backend virtual environment...
if not exist "%VENV_PYTHON%" (
  python -m venv "%ROOT%backend\.venv"
  if errorlevel 1 (
    echo [ERROR] Failed to create backend virtual environment.
    goto :fail
  )
)

echo [3/7] Updating the virtual environment installer...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.
  goto :fail
)

echo [4/7] Installing and verifying core backend dependencies...
"%VENV_PYTHON%" -m pip install -r "%ROOT%backend\requirements.txt"
if errorlevel 1 (
  echo [ERROR] Failed to install core backend dependencies.
  goto :fail
)
"%VENV_PYTHON%" -c "import importlib.util as u, sys; mods=['fastapi','uvicorn','websockets','loguru','pydantic','yaml','numpy','soundfile','httpx','requests','aiohttp','openai','anthropic','edge_tts','pysbd','langdetect','pydub','sherpa_onnx','onnxruntime','multipart','chardet','jinja2','tqdm','mcp','letta_client','win32crypt']; missing=[m for m in mods if u.find_spec(m) is None]; print('Missing core dependencies: '+', '.join(missing)) if missing else print('Core backend dependencies verified.'); raise SystemExit(1 if missing else 0)"
if errorlevel 1 goto :fail
"%VENV_PYTHON%" "%ROOT%backend\check_runtime_dependencies.py"
if errorlevel 1 (
  echo [ERROR] Core dependency versions are incompatible.
  goto :fail
)
"%VENV_PYTHON%" -c "import sys; sys.path.insert(0, r'%ROOT%backend'); from src.open_llm_vtuber.service_context import ServiceContext; from src.open_llm_vtuber.websocket_handler import WebSocketHandler; print('Core backend imports verified.')"
if errorlevel 1 (
  echo [ERROR] Core backend import verification failed.
  goto :fail
)

echo [5/7] Optional OmniVoice voice cloning...
if /I "%MELOMATE_VOICE_CLONE%"=="1" set "OPTIONAL_REQUESTED=1"
if /I "%MELOMATE_VOICE_CLONE%"=="yes" set "OPTIONAL_REQUESTED=1"
if /I "%MELOMATE_VOICE_CLONE%"=="0" goto :optional_decided
if /I "%MELOMATE_VOICE_CLONE%"=="no" goto :optional_decided
if defined MELOMATE_VOICE_CLONE (
  echo [ERROR] MELOMATE_VOICE_CLONE must be 1, yes, 0, or no.
  goto :fail
)

echo Voice cloning downloads large AI packages and models.
choice /C YN /N /M "Install voice cloning now? [Y/N]: "
if errorlevel 2 (
  set "OPTIONAL_REQUESTED=0"
) else (
  set "OPTIONAL_REQUESTED=1"
)

:optional_decided
if "%OPTIONAL_REQUESTED%"=="0" (
  echo Voice cloning was skipped. Normal MeloMate voice chat will remain available.
  goto :build_frontend
)

if /I "%MELOMATE_VOICE_CLONE_DEVICE%"=="gpu" set "VOICE_DEVICE=gpu"
if /I "%MELOMATE_VOICE_CLONE_DEVICE%"=="cpu" set "VOICE_DEVICE=cpu"
if defined MELOMATE_VOICE_CLONE_DEVICE (
  if /I not "%MELOMATE_VOICE_CLONE_DEVICE%"=="gpu" if /I not "%MELOMATE_VOICE_CLONE_DEVICE%"=="cpu" (
    echo [ERROR] MELOMATE_VOICE_CLONE_DEVICE must be gpu or cpu.
    goto :fail
  )
  goto :install_voice_clone
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo No NVIDIA driver was detected. CPU voice cloning will be installed.
  set "VOICE_DEVICE=cpu"
) else (
  choice /C GC /N /M "NVIDIA GPU detected. Install [G]PU acceleration or [C]PU mode? [G/C]: "
  if errorlevel 2 (set "VOICE_DEVICE=cpu") else (set "VOICE_DEVICE=gpu")
)

:install_voice_clone
if /I "%VOICE_DEVICE%"=="gpu" (
  echo Installing matching CUDA 12.8 PyTorch packages...
  "%VENV_PYTHON%" -m pip install --force-reinstall -r "%ROOT%backend\requirements-gpu-cu128.txt"
) else (
  echo Installing matching CPU PyTorch packages...
  "%VENV_PYTHON%" -m pip install --force-reinstall torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 (
  echo [ERROR] Failed to install the matching PyTorch packages.
  set "OPTIONAL_FAILED=1"
  goto :build_frontend
)

"%VENV_PYTHON%" -m pip install -r "%ROOT%backend\omnivoice-requirements.txt"
if errorlevel 1 (
  echo [ERROR] Failed to install OmniVoice dependencies.
  set "OPTIONAL_FAILED=1"
  goto :build_frontend
)

"%VENV_PYTHON%" -c "import torch, torchaudio, transformers, accelerate, librosa, omnivoice, huggingface_hub; tv=torch.__version__.split('+')[0]; av=torchaudio.__version__.split('+')[0]; print('torch:', torch.__version__); print('torchaudio:', torchaudio.__version__); print('voice cloning dependencies verified.'); raise SystemExit(0 if tv == av else 1)"
if errorlevel 1 (
  echo [ERROR] Voice cloning dependency verification failed or PyTorch versions do not match.
  set "OPTIONAL_FAILED=1"
  goto :build_frontend
)
if /I "%VOICE_DEVICE%"=="gpu" (
  "%VENV_PYTHON%" -c "import torch; print('cuda available:', torch.cuda.is_available()); print('cuda device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
  if errorlevel 1 (
    echo [ERROR] CUDA PyTorch was installed, but the NVIDIA GPU is not available.
    echo Update the NVIDIA driver or rerun setup-windows.bat and choose CPU mode.
    set "OPTIONAL_FAILED=1"
    goto :build_frontend
  )
)

call "%ROOT%download-omnivoice-model.bat" --no-pause
if errorlevel 1 (
  echo [ERROR] Voice cloning packages are installed, but model download failed.
  set "OPTIONAL_FAILED=1"
)

:build_frontend
echo [6/7] Building the frontend...
call npm run build
if errorlevel 1 (
  echo [ERROR] Frontend production build failed.
  goto :fail
)

echo [7/7] Performing final installation checks...
if not exist "%ROOT%dist\index.html" (
  echo [ERROR] dist\index.html was not produced.
  goto :fail
)
"%VENV_PYTHON%" -m pip check
if errorlevel 1 (
  echo [WARN] pip found a dependency conflict. Core imports were already verified above.
  if "%OPTIONAL_REQUESTED%"=="1" set "OPTIONAL_FAILED=1"
)

echo.
if "%OPTIONAL_FAILED%"=="1" (
  echo Core MeloMate setup finished successfully.
  echo Voice cloning was requested but is not ready. Review the error above, then run setup-windows.bat again.
  echo Normal voice chat can still be used with start.bat.
  goto :optional_fail
)

echo Setup finished successfully.
if "%OPTIONAL_REQUESTED%"=="1" (
  echo Voice cloning and its models are ready in %VOICE_DEVICE% mode.
) else (
  echo Voice cloning was not installed. You can rerun this same script later to add it.
)
echo Configure backend\conf.yaml if needed, then run start.bat.
if not "%MELOMATE_NO_PAUSE%"=="1" pause
exit /b 0

:optional_fail
if not "%MELOMATE_NO_PAUSE%"=="1" pause
exit /b 2

:fail
echo.
echo Setup failed. Read the error message above.
if not "%MELOMATE_NO_PAUSE%"=="1" pause
exit /b 1
