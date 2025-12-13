import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
import time

def get_nba_data():
    print("--- Demarrage de la recuperation des donnees NBA (Version Four Factors) ---")
    
    try:
        # On récupère les données brutes
        gamefinder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable='00',
            season_type_nullable='Regular Season',
            timeout=60
        )
        
        # Transformation en tableau
        games = gamefinder.get_data_frames()[0]
        
        # Nettoyage des dates
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        
        # On garde l'historique récent (2023+)
        games = games[games['GAME_DATE'] > '2023-01-01']
        games = games.sort_values('GAME_DATE')
        
        # VÉRIFICATION
        required_cols = ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'MATCHUP', 'WL', 
                         'PTS', 'FGM', 'FGA', 'FG3M', 'TOV', 'OREB', 'FTM', 'FTA']
        
        missing = [col for col in required_cols if col not in games.columns]
        if missing:
            print(f"[ATTENTION] Il manque ces colonnes vitales : {missing}")
        else:
            print(f"[OK] Toutes les stats 'Four Factors' sont presentes.")

        print(f"Succes ! {len(games)} matchs recuperes.")
        
        # Sauvegarde
        games.to_csv('nba_games.csv', index=False)
        print("Fichier 'nba_games.csv' mis a jour.")
        
    except Exception as e:
        print("\n[ERREUR] Probleme de connexion :")
        print(e)
        exit(1)

if __name__ == "__main__":
    get_nba_data()