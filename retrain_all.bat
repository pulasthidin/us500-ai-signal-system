@echo off
title US500 Manual Retrain — M1, M2, M3
cd /d "%~dp0"

echo ============================================
echo   US500 Manual Retrain
echo   All 3 Models (M1 + M2 + M3)
echo ============================================
echo.

python scripts\retrain_all_models.py

echo.
echo ============================================
echo   Done. Restart the app to load new models.
echo ============================================
echo.
pause
