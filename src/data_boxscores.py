import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2
import time
import os

# --- CONFIG ---
HISTORY_FILE = 'data/bets_history.csv'
OUTPUT_FILE = 'data/boxscores.csv'

def fetch_all_boxscores():
    print("--- RECUPERATION DES BOX SCORES ---")
    
    if not os.path.exists(HISTORY_FILE):
        print("Pas d'historique trouvé.")
        return

    # 1. Charger l'historique
    hist = pd.read_csv(HISTORY_FILE)
    
    # 2. Filtrer les matchs terminés (Gagné/Perdu)
    finished = hist[hist['Result'].isin(['GAGNE', 'PERDU'])].copy()
    print(f"{len(finished)} matchs terminés trouvés dans l'historique.")

    # 3. Charger les boxscores existants (pour ne pas refaire le travail)
    existing_ids = []
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        if 'GAME_ID' in existing_df.columns:
            existing_ids = existing_df['GAME_ID'].astype(str).unique().tolist()
            print(f"{len(existing_ids)} boxscores déjà en base.")
    
    all_stats = []
    
    # 4. Boucle de récupération (avec pause pour éviter le ban API)
    count = 0
    for _, row in finished.iterrows():
        # Il faut retrouver l'ID du match. 
        # Astuce : On ne l'a pas dans le CSV historique (v5/v6), mais on l'a dans le fichier games_ready ou on peut le chercher.
        # Pour faire simple ici, on va supposer qu'on utilise le LeagueGameFinder pour retrouver l'ID via la date et les équipes.
        # C'est complexe sans l'ID.
        
        # PROPOSITION SIMPLIFIÉE :
        # Pour l'instant, ce script est une ébauche. 
        # L'idéal serait d'ajouter la colonne 'GAME_ID' dans 'bets_history.csv' lors du scan quotidien.
        pass 
        
    print("NOTE : Pour que ce script fonctionne parfaitement, il faudrait sauvegarder le GAME_ID dans bets_history.csv")
    print("C'est une évolution pour la v10.")

if __name__ == "__main__":
    fetch_all_boxscores()