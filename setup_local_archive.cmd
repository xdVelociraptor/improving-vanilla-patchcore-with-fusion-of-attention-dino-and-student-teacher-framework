@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_local_archive.ps1"
if errorlevel 1 (
  echo.
  echo Setup stopped. Read the message above and press any key to close.
  pause >nul
  exit /b 1
)
echo.
echo Setup completed. Press any key to close.
pause >nul
