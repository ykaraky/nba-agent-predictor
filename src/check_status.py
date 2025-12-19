import sys
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2
import pandas as pd

def check_nba_status():
    # 1. Date cible : Hier (car on veut les résultats de la nuit dernière)
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y-%m-%d')
    date_disp = yesterday.strftime('%d.%m.%Y')
    
    print(f"--- CONTROLE MATCHS DU {date_disp} ---")
    print("Connexion API en cours...")

    try:
        # 2. Interrogation Scoreboard
        board = scoreboardv2.ScoreboardV2(game_date=date_str)
        games = board.game_header.get_data_frame()

        if games.empty:
            print(f"[INFO] Aucun match n'etait prevu hier.")
            print(">> Tu peux lancer la mise a jour.")
            return

        # 3. Analyse des statuts (Status ID 3 = Final)
        total_games = len(games)
        finished_games = len(games[games['GAME_STATUS_ID'] == 3])
        
        print(f"Matchs trouves  : {total_games}")
        print(f"Matchs termines : {finished_games}")
        print("-" * 30)

        if finished_games == total_games and total_games > 0:
            print("[ OK ] TOUT EST TERMINE.")
            print(">> FEU VERT : Tu peux lancer GO_NBA.bat")
        elif finished_games == 0:
            print("[ ATTENTE ] AUCUN RESULTAT ENCORE DISPO.")
            print(">> FEU ROUGE : Trop tot. Prends un cafe.")
        else:
            diff = total_games - finished_games
            print(f"[ ATTENTION ] IL MANQUE {diff} MATCH(S).")
            print(">> FEU ORANGE : Attends encore 15-20 min.")

    except Exception as e:
        print(f"[ERREUR] Impossible de joindre l'API : {e}")

if __name__ == "__main__":
    check_nba_status()