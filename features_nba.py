import pandas as pd
import numpy as np
import os

print("--- Calcul des FOUR FACTORS ---")

# Vérification d'accès aux fichiers
if os.path.exists('nba_games_ready.csv'):
    try:
        # On essaie d'ouvrir le fichier en écriture juste pour voir si on peut
        f = open('nba_games_ready.csv', 'a')
        f.close()
    except PermissionError:
        print("[ERREUR] Le fichier 'nba_games_ready.csv' est ouvert dans Excel !")
        print("Fermez-le et relancez.")
        exit(1)

try:
    # 1. Chargement
    if not os.path.exists('nba_games.csv'):
        print("[ERREUR] 'nba_games.csv' introuvable.")
        exit(1)
        
    df = pd.read_csv('nba_games.csv')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])

    # --- CALCUL DES FOUR FACTORS ---
    # On évite les divisions par zéro avec numpy
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, np.nan)
    df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV']).replace(0, np.nan)
    df['FT_RATE'] = df['FTM'] / df['FGA'].replace(0, np.nan)
    df['ORB_RAW'] = df['OREB']

    # --- MOYENNES GLISSANTES ---
    factors = ['EFG_PCT', 'TOV_PCT', 'FT_RATE', 'ORB_RAW', 'WIN']
    print("Calcul des moyennes...")

    for factor in factors:
        new_col_name = f"{factor}_LAST_5"
        df[new_col_name] = df.groupby('TEAM_ID')[factor].transform(
            lambda x: x.shift(1).rolling(window=5).mean()
        )

    # --- FATIGUE ---
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(upper=7)

    # Nettoyage
    df_final = df.dropna(subset=[f"{f}_LAST_5" for f in factors])

    # Sauvegarde
    df_final.to_csv('nba_games_ready.csv', index=False)
    print(f"[OK] Termine. {len(df_final)} lignes generees.")

except Exception as e:
    print("[ERREUR CRITIQUE] :")
    print(e)
    exit(1)