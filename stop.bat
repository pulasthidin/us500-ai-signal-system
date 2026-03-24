@echo off
title Stop US500 Signal App
echo Stopping US500 Signal App...
echo.
taskkill /f /im python.exe /fi "WINDOWTITLE eq US500 Signal App" >nul 2>&1
if errorlevel 1 (
    echo App was not running.
) else (
    echo App stopped successfully.
)
echo.
pause
