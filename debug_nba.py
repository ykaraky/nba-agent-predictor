from nba_api.stats.endpoints import scoreboardv2
import pandas as pd

# Date du 23 Décembre (où il y a eu des matchs)
TARGET_DATE = "2025-12-23"

print(f"--- TEST API NBA SUR {TARGET_DATE} ---")

try:
    # Appel API
    board = scoreboardv2.ScoreboardV2(game_date=TARGET_DATE)
    header = board.game_header.get_data_frame()
    scores = board.line_score.get_data_frame()

    print(f"Matchs trouvés (Header) : {len(header)}")
    print(f"Lignes de scores trouvées : {len(scores)}")
    
    if not scores.empty:
        print("\n--- ECHANTILLON DES SCORES ---")
        # On affiche les colonnes critiques
        subset = scores[['GAME_ID', 'TEAM_ID', 'PTS']]
        print(subset.head(10))
        
        print("\n--- TYPES DES DONNEES ---")
        print(f"Type GAME_ID: {type(scores.iloc[0]['GAME_ID'])}")
        print(f"Exemple brut GAME_ID: '{scores.iloc[0]['GAME_ID']}'")
        print(f"Type PTS: {type(scores.iloc[0]['PTS'])}")
        print(f"Exemple brut PTS: '{scores.iloc[0]['PTS']}'")
    else:
        print("!!! AUCUN SCORE RENVOYÉ PAR L'API !!!")

except Exception as e:
    print(f"ERREUR CRITIQUE : {e}")

input("\nAppuie sur Entrée pour quitter...")