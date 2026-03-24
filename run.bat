@echo off
title US500 Signal App
cd /d "%~dp0"
echo Starting US500 Signal App...
echo.
echo Do not close this window.
echo Telegram will confirm when ready.
echo.
set PYTHONIOENCODING=utf-8
python live.py

REM If app crashes show error
if errorlevel 1 (
    echo.
    echo ========================================
    echo App stopped unexpectedly.
    echo Check logs folder for details.
    echo ========================================
    pause
)
