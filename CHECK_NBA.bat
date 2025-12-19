@echo off
title NBA STATUS CHECKER
color 0E

echo ==========================================
echo      VERIFICATION API NBA
echo ==========================================
echo.

:: 1. Se placer dans le dossier du script
cd /d "%~dp0"

:: 2. Lancer le script de verification
python src/check_status.py

echo.
echo ==========================================
pause