@echo off
REM stop.bat - Stop CT Viewer server (Windows)

echo Stopping CT Viewer...

REM Find and kill Python processes running ct_viewer.py
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| find "PID:"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | find "ct_viewer.py" >nul
    if not errorlevel 1 (
        echo Stopping process %%a...
        taskkill /f /pid %%a >nul 2>&1
    )
)

echo Done.
pause
