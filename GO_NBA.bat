@echo off
title NBA MANAGER - AUTO LAUNCHER
color 0A

echo ==========================================
echo      DEMARRAGE DE L'AGENT NBA...
echo ==========================================
echo.

:: 1. Se placer dans le dossier du script (CRUCIAL)
cd /d "%~dp0"

:: 2. Lancer la routine python (qui fait Data + Calculs + Git + Streamlit)
python daily_routine.py

:: 3. Si ça plante, on ne ferme pas la fenêtre tout de suite
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERREUR] Une erreur s'est produite. Regarde au-dessus.
    pause
) else (
    echo.
    echo [INFO] Interface fermee.
    pause
)