import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# Fonction qui va chercher les matchs
def get_nba_data():
    print("--- Démarrage de la récupération des données NBA ---")
    print("Connexion à l'API en cours...")
    
    # On demande les matchs de la NBA (League ID '00') pour la saison régulière
    # On ne filtre pas par saison ici pour en récupérer un maximum par défaut
    gamefinder = leaguegamefinder.LeagueGameFinder(
        league_id_nullable='00',
        season_type_nullable='Regular Season' 
    )
    
    # Transformation en tableau (DataFrame)
    games = gamefinder.get_data_frames()[0]
    
    # Petit nettoyage : on met la date au bon format
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    
    # On garde les matchs à partir de 2023 pour que ce soit plus rapide à lire
    # (Sinon il remonte très loin dans l'histoire)
    games = games[games['GAME_DATE'] > '2023-01-01']
    
    # On trie du plus vieux au plus récent
    games = games.sort_values('GAME_DATE')
    
    return games

# --- Lancement du script ---

try:
    df = get_nba_data()
    
    print("\nSUCCÈS ! Données récupérées.")
    print(f"Nombre de lignes trouvées : {len(df)}")
    print("\nVoici les 5 derniers matchs récupérés :")
    print(df[['GAME_DATE', 'MATCHUP', 'WL', 'PTS']].tail(5))
    
    # Sauvegarde
    df.to_csv('nba_games.csv', index=False)
    print("\nFichier 'nba_games.csv' créé dans ton dossier.")

except Exception as e:
    print("\nERREUR :")
    print(e)
    print("Vérifie ta connexion internet.")