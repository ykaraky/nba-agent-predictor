import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.http import NBAStatsHTTP
import time

# --- ASTUCE ANTI-ROBOT ---
# On modifie l'identité du script pour qu'il ressemble à un navigateur humain
# Cela évite de se faire bloquer par les serveurs de la NBA
nba_headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nba.com/'
}

# Appliquer ces headers à la librairie nba_api
NBAStatsHTTP.headers = nba_headers

def get_nba_data():
    print("--- Démarrage de la récupération des données NBA ---")
    
    # On va essayer 3 fois de suite (Retry Logic)
    max_retries = 3
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Tentative de connexion n°{attempt}...")
            
            # timeout=60 : On laisse 60 secondes au serveur pour répondre (au lieu de 30)
            gamefinder = leaguegamefinder.LeagueGameFinder(
                league_id_nullable='00',
                season_type_nullable='Regular Season',
                timeout=60 
            )
            
            # Transformation en tableau (DataFrame)
            games = gamefinder.get_data_frames()[0]
            
            # Si on arrive ici, c'est que ça a marché !
            print(f"Succès ! {len(games)} matchs récupérés.")
            
            # Nettoyage
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games = games[games['GAME_DATE'] > '2023-01-01']
            games = games.sort_values('GAME_DATE')
            
            # Sauvegarde
            games.to_csv('nba_games.csv', index=False)
            print("Fichier 'nba_games.csv' sauvegardé.")
            
            return games # On quitte la fonction, tout est bon
            
        except Exception as e:
            print(f"⚠️ Échec de la tentative {attempt} : {e}")
            if attempt < max_retries:
                print("Pause de 10 secondes avant de réessayer...")
                time.sleep(10)
            else:
                print("❌ Abandon après 3 échecs.")
                # Important : On relance l'erreur pour que GitHub sache que ça a planté
                raise e

# --- Lancement ---
if __name__ == "__main__":
    get_nba_data()