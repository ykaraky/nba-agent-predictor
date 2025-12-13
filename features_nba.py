import pandas as pd
import numpy as np

print("--- Calcul des FOUR FACTORS (Niveau Expert) ---")

try:
    # 1. Chargement
    df = pd.read_csv('nba_games.csv')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])

    # --- CALCUL DES FOUR FACTORS BRUTS PAR MATCH ---
    # 1. Effective Field Goal %
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']

    # 2. Turnover %
    df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])

    # 3. Free Throw Rate
    df['FT_RATE'] = df['FTM'] / df['FGA']

    # 4. Offensive Rebound % (Brut)
    df['ORB_RAW'] = df['OREB']

    # --- CALCUL DES MOYENNES GLISSANTES (5 MATCHS) ---
    factors = ['EFG_PCT', 'TOV_PCT', 'FT_RATE', 'ORB_RAW', 'WIN']

    print("Calcul des moyennes glissantes sur les Four Factors...")

    for factor in factors:
        new_col_name = f"{factor}_LAST_5"
        df[new_col_name] = df.groupby('TEAM_ID')[factor].transform(
            lambda x: x.shift(1).rolling(window=5).mean()
        )

    # --- CALCUL DE LA FATIGUE ---
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(upper=7)

    # Nettoyage final
    df_final = df.dropna(subset=[f"{f}_LAST_5" for f in factors])

    # Sauvegarde
    df_final.to_csv('nba_games_ready.csv', index=False)
    # J'ai remplacé le ✅ par [OK]
    print(f"[OK] Termine ! {len(df_final)} matchs prets avec les Four Factors.")

except Exception as e:
    print("[ERREUR] dans features_nba.py :")
    print(e)
    exit(1)