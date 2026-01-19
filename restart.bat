@echo off
REM restart.bat - Restart CT Viewer server (Windows)

echo Restarting CT Viewer...
echo.

REM Stop existing server
call "%~dp0stop.bat"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start server
call "%~dp0start.bat"
