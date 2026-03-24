@echo off
title US500 App Status
echo ========================================
echo US500 Signal App Status
echo ========================================
echo.
tasklist /fi "WINDOWTITLE eq US500 Signal App" 2>nul | find "python.exe" >nul
if errorlevel 1 (
    echo Status: NOT RUNNING
    echo Double click run.bat to start.
) else (
    echo Status: RUNNING
)
echo.
echo Last heartbeat:
if exist "data\heartbeat.json" (
    type data\heartbeat.json
) else (
    echo No heartbeat file found.
)
echo.
echo.
echo Last 5 log lines:
if exist "logs\app.log" (
    powershell -command "Get-Content logs\app.log -Tail 5"
) else (
    echo No log file found.
)
echo.
pause
