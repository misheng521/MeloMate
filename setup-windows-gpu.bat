@echo off
setlocal

chcp 65001 >nul
set "ROOT=%~dp0"
echo This compatibility shortcut now uses the single MeloMate installer.
echo It preselects OmniVoice with NVIDIA CUDA 12.8 acceleration.
echo.

set "MELOMATE_VOICE_CLONE=1"
set "MELOMATE_VOICE_CLONE_DEVICE=gpu"
set "MELOMATE_NO_PAUSE=1"
call "%ROOT%setup-windows.bat"
set "SETUP_RESULT=%ERRORLEVEL%"

if not "%SETUP_RESULT%"=="0" (
  echo.
  echo [ERROR] GPU setup did not complete. See the error above.
  pause
  exit /b %SETUP_RESULT%
)

echo.
echo GPU setup finished. Run start.bat.
pause
exit /b 0
