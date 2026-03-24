@echo off
title US500 App Setup
echo ========================================
echo US500 Signal App - First Time Setup
echo ========================================
echo.

REM Check Python installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install Python 3.11+ from python.org
    echo Make sure to check "Add to PATH" during install.
    echo.
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo Installing packages (this may take a few minutes)...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup complete.
echo ========================================
echo.
echo Next steps:
echo   1. Open .env file in a text editor
echo   2. Replace all placeholder values with your real API keys
echo   3. Double click run.bat to start the app
echo.

echo.
echo Do you want the app to start automatically
echo when Windows starts? (Y/N)
set /p choice=
if /i "%choice%"=="Y" (
    set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
    echo @echo off > "%STARTUP%\US500App.bat"
    echo cd /d "%~dp0" >> "%STARTUP%\US500App.bat"
    echo call run.bat >> "%STARTUP%\US500App.bat"
    echo.
    echo Added to Windows startup.
    echo App will start automatically on boot.
) else (
    echo.
    echo Skipped auto-start.
    echo You can always double click run.bat manually.
)
echo.
pause
