@echo off
REM start.bat - Start CT Viewer on Windows
REM Double-click this file to launch the CT Viewer

title CT Viewer
cd /d "%~dp0"

echo.
echo ========================================
echo         CT VIEWER FOR WINDOWS
echo ========================================
echo.

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo  Python is not installed on this computer.
    echo.
    echo  To install Python:
    echo    1. Press any key to open the download page
    echo    2. Click the yellow "Download Python" button
    echo    3. Run the installer
    echo    4. IMPORTANT: Check "Add python.exe to PATH"
    echo    5. Click "Install Now"
    echo    6. After install, run this file again
    echo.
    echo ========================================
    pause
    start "" "https://www.python.org/downloads/"
    exit /b 1
)

echo  Python found:
python --version
echo.

REM Check if packages are installed by trying to import flask
python -c "import flask" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo  Installing required packages...
    echo  (This only happens once, please wait)
    echo.
    pip install flask pillow numpy scipy scikit-image pydicom pylibjpeg
    if %ERRORLEVEL% neq 0 (
        echo.
        echo  ERROR: Could not install packages.
        echo  Please check your internet connection.
        pause
        exit /b 1
    )
    echo.
    echo  Packages installed successfully!
    echo.
)

echo  Starting CT Viewer...
echo.
echo  Opening browser to: http://localhost:7002
echo.
echo  ----------------------------------------
echo  To STOP the viewer: Close this window
echo                 or press Ctrl+C
echo  ----------------------------------------
echo.

REM Open browser
start "" http://localhost:7002

REM Small delay to let browser open first
timeout /t 2 /nobreak >nul

REM Run the server
python ct_viewer.py

pause
