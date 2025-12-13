import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
import time

def get_nba_data():
    print("--- Démarrage de la récupération des données NBA (Mode Simple) ---")
    
    try:
        # On demande gentiment les données sans headers complexes
        # timeout=60 laisse le temps de charger si ta connexion rame un peu
        gamefinder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable='00',
            season_type_nullable='Regular Season',
            timeout=60
        )
        
        # Transformation en tableau
        games = gamefinder.get_data_frames()[0]
        
        # Nettoyage
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        # On garde l'historique récent (2023+)
        games = games[games['GAME_DATE'] > '2023-01-01']
        games = games.sort_values('GAME_DATE')
        
        print(f"Succès ! {len(games)} matchs récupérés.")
        
        # Sauvegarde
        games.to_csv('nba_games.csv', index=False)
        print("Fichier 'nba_games.csv' mis à jour.")
        
    except Exception as e:
        print("\n❌ Erreur de connexion :")
        print(e)
        print("Conseil : Si l'erreur persiste (WinError 10054), attends 10 minutes et réessaie.")
        # On arrête le script ici pour ne pas lancer la suite avec des données pourries
        exit(1)

if __name__ == "__main__":
    get_nba_data()