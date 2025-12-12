import pandas as pd
import numpy as np

print("--- Traitement des données + FATIGUE ---")

# 1. Chargement
try:
    df = pd.read_csv('nba_games.csv')
except FileNotFoundError:
    print("Erreur : 'nba_games.csv' introuvable.")
    exit()

df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df['WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)

# Tri indispensable pour calculer le temps entre deux matchs
df = df.sort_values(by=['TEAM_ID', 'GAME_DATE'])

# --- NOUVEAUTÉ : CALCUL DES JOURS DE REPOS ---
print("Calcul de la fatigue (Rest Days)...")

# On décale la date d'une ligne pour avoir la date du match précédent
df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)

# On fait la soustraction : Date du match - Date du match précédent
df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days

# Gestion des cas particuliers (début de saison)
# Si c'est le premier match, on met 3 jours de repos par défaut (frais)
df['DAYS_REST'] = df['DAYS_REST'].fillna(3)

# On plafonne à 7 jours (pour éviter de fausser les stats avec la pause All-Star)
df['DAYS_REST'] = df['DAYS_REST'].clip(upper=7)

# --- FIN NOUVEAUTÉ ---

# Calcul des moyennes glissantes (comme avant)
cols_to_average = ['PTS', 'PLUS_MINUS', 'WIN']

print("Calcul des moyennes glissantes...")
for col in cols_to_average:
    new_col_name = f"{col}_LAST_5"
    df[new_col_name] = df.groupby('TEAM_ID')[col].transform(
        lambda x: x.shift(1).rolling(window=5).mean()
    )

df_final = df.dropna()

# On sauvegarde
df_final.to_csv('nba_games_ready.csv', index=False)
print("\nTransformation terminée ! La colonne 'DAYS_REST' a été ajoutée.")
print(df_final[['GAME_DATE', 'MATCHUP', 'DAYS_REST']].tail(5))