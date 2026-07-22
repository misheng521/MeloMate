@echo off
setlocal

chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo This setup installs MeloMate plus CUDA 12.8 PyTorch for NVIDIA GPU voice cloning.
echo If your computer does not have an NVIDIA GPU, use setup-windows.bat instead.
echo.

call "%ROOT%setup-windows.bat"
if errorlevel 1 (
  echo [ERROR] Base setup failed.
  pause
  exit /b 1
)

echo.
echo Installing CUDA PyTorch packages for OmniVoice...
"%ROOT%backend\.venv\Scripts\python.exe" -m pip install --force-reinstall -r "%ROOT%backend\requirements-gpu-cu128.txt"
if errorlevel 1 (
  echo [ERROR] Failed to install CUDA PyTorch packages.
  echo If this computer does not support CUDA 12.8, use setup-windows.bat instead.
  pause
  exit /b 1
)

echo.
echo Checking GPU availability...
"%ROOT%backend\.venv\Scripts\python.exe" -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu count:', torch.cuda.device_count()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if errorlevel 1 (
  echo [ERROR] GPU check failed.
  pause
  exit /b 1
)

echo.
echo GPU setup finished. Run start.bat.
pause
exit /b 0
