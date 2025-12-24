@echo off
title RECUPERATION DONNEES PROD
color 0B

echo ==========================================
echo      IMPORT DES DONNEES DEPUIS MAIN
echo ==========================================
echo.
echo Vous etes sur le point d'ecraser le fichier CSV local
echo avec la version la plus recente de la branche MAIN.
echo.
echo Cela permet d'avoir les derniers paris a jour sur votre branche de DEV.
echo.
pause

:: 1. On recupere les infos du depot (sans changer de branche)
git fetch origin main

:: 2. On ecrase le fichier local par celui de main
git checkout origin/main -- data/bets_history.csv

if %errorlevel% neq 0 (
    color 0C
    echo [ERREUR] Impossible de recuperer le fichier.
) else (
    color 0A
    echo [SUCCES] Fichier bets_history.csv mis a jour depuis MAIN !
    echo.
    echo Vous pouvez travailler sur votre branche avec des donnees fraiches.
)

pause