@echo off
title US500 App Logs
echo ========================================
echo US500 Signal App - Recent Logs
echo ========================================
echo.
if exist "logs\app.log" (
    type logs\app.log | more
) else (
    echo No log file found.
    echo Run the app first to generate logs.
)
echo.
pause
