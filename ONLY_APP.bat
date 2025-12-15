@echo off
title NBA MANAGER - INTERFACE LOCALE
color 0B

echo ==========================================
echo      LANCEMENT DE L'INTERFACE SEULE
echo ==========================================
echo [INFO] Pas de mise a jour, pas de synchro Cloud.
echo.

cd /d "%~dp0"

:: Lance uniquement Streamlit
python -m streamlit run app.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERREUR] Impossible de lancer l'interface.
    pause
)