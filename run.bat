@echo off
title US500 Signal App
cd /d "%~dp0"
echo Starting US500 Signal App...
echo.
echo Do not close this window.
echo Telegram will confirm when ready.
echo.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
"C:\Users\Pulasthi.Ranathunga\AppData\Local\Programs\Python\Python314\python.exe" live.py

REM If app crashes show error
if errorlevel 1 (
    echo.
    echo ========================================
    echo App stopped unexpectedly.
    echo Check logs folder for details.
    echo ========================================
    pause
)
