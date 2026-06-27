@echo off
echo ============================================
echo  DualVision AI Detector - Build Script
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.12 from python.org
    pause
    exit /b 1
)

REM Check PyInstaller
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo [1/3] Cleaning previous build...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo [2/3] Building executable...
pyinstaller DualVisionAI.spec

echo [3/3] Done!
if exist dist\DualVisionAI\DualVisionAI.exe (
    echo.
    echo [SUCCESS] Executable created at: dist\DualVisionAI\DualVisionAI.exe
) else (
    echo [WARNING] Build may have issues. Check above output.
)

echo.
pause
