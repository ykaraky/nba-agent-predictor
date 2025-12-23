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

:: 3. Synchroniser vers le Cloud (Supabase)
python src/sync_cloud.py

:: 4. Si ça plante, on ne ferme pas la fenêtre tout de suite
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERREUR] Une erreur s'est produite. Regarde au-dessus.
    pause
) else (
    echo.
    echo [INFO] Tout est a jour (Local + Cloud).
    pause
)