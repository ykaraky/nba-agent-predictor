import pandas as pd
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime

# Date d'aujourd'hui (selon ton système)
today = datetime.now().strftime('%Y-%m-%d')
print(f"--- DIAGNOSTIC API POUR LE {today} ---")

try:
    # On appelle l'API
    board = scoreboardv2.ScoreboardV2(game_date=today)
    
    # On regarde le premier tableau (GameHeader)
    df = board.game_header.get_data_frame()
    
    print(f"Nombre de lignes brutes : {len(df)}")
    
    if len(df) > 0:
        print("\n--- LISTE DES COLONNES ---")
        print(list(df.columns))
        
        print("\n--- CONTENU DE LA PREMIÈRE LIGNE ---")
        # On affiche la première ligne transposée pour bien lire
        print(df.iloc[0])
        
        print("\n--- ESSAI DE LECTURE DES IDs ---")
        print(df[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']])
    else:
        print("Le tableau est vide.")

except Exception as e:
    print("Erreur critique :")
    print(e)

input("\nAppuie sur Entrée pour fermer.")