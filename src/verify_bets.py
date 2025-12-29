import pandas as pd
import os
import time
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

HISTORY_FILE = 'data/bets_history.csv'

# --- OUTILS ---
def get_teams_dict():
    nba_teams = teams.get_teams()
    return {t['id']: {'full': t['full_name'], 'code': t['abbreviation']} for t in nba_teams}

TEAMS_DB = get_teams_dict()

def clean_id(val):
    """Nettoyage ID robuste (int -> string sans zero)"""
    try: return str(int(float(val))).lstrip('0')
    except: return str(val).lstrip('0')

def verify():
    print("\n--- VÉRIFICATION DES RÉSULTATS (LIVE API) ---")
    
    if not os.path.exists(HISTORY_FILE):
        print("[ERREUR] Pas d'historique.")
        return

    df = pd.read_csv(HISTORY_FILE)
    
    # 1. Identifier les matchs sans résultat (ou marqués 'En attente')
    # On regarde si Real_Winner est vide ou null
    mask_pending = df['Real_Winner'].isna() | (df['Real_Winner'] == "") | (df['Real_Winner'] == "En attente...")
    
    # On ne garde que les dates passées (pour ne pas vérifier les matchs futurs)
    today_str = datetime.now().strftime('%Y-%m-%d')
    mask_date = df['Date'] < today_str
    
    pending_indices = df[mask_pending & mask_date].index
    
    if len(pending_indices) == 0:
        print("[INFO] Aucun match passé en attente de résultat.")
        return

    print(f"[INFO] {len(pending_indices)} matchs à vérifier...")
    
    # Récupérer les dates uniques concernées pour ne pas spammer l'API
    dates_to_check = df.loc[pending_indices, 'Date'].unique()
    
    updates = 0
    
    for d_str in dates_to_check:
        print(f"   -> Scan API pour le {d_str}...")
        try:
            # Format Date US pour l'API
            d_us = datetime.strptime(d_str, '%Y-%m-%d').strftime('%m/%d/%Y')
            
            finder = leaguegamefinder.LeagueGameFinder(
                date_from_nullable=d_us,
                date_to_nullable=d_us,
                league_id_nullable='00'
            )
            results = finder.get_data_frames()[0]
            
            if results.empty:
                print("      (Pas de données API)")
                continue

            # Création d'un dico des gagnants pour cette date
            # Key: GameID, Value: WinnerName
            winners_map = {}
            
            # On groupe par GameID pour trouver qui a gagné
            games_groups = results.groupby('GAME_ID')
            
            for gid, rows in games_groups:
                # On cherche la ligne avec WL = 'W'
                winner_row = rows[rows['WL'] == 'W']
                if not winner_row.empty:
                    w_team_id = int(winner_row.iloc[0]['TEAM_ID'])
                    # On traduit l'ID en Nom Complet (comme dans le CSV)
                    if w_team_id in TEAMS_DB:
                        w_name = TEAMS_DB[w_team_id]['full']
                        gid_clean = clean_id(gid)
                        winners_map[gid_clean] = w_name

            # Mise à jour du DataFrame
            # Pour chaque ligne du CSV à cette date, on essaie de trouver le vainqueur
            # Problème : Le CSV n'a pas le GameID. On doit matcher par Noms d'équipe.
            
            # On construit un mapping Nom -> Resultat depuis l'API pour cette date
            # Dict: {NomTeam: "Gagné/Perdu"}
            team_results_day = {}
            for _, r in results.iterrows():
                tid = int(r['TEAM_ID'])
                if tid in TEAMS_DB:
                    tname = TEAMS_DB[tid]['full']
                    outcome = "W" if r['WL'] == 'W' else "L"
                    team_results_day[tname] = outcome

            # Application au CSV
            for idx in pending_indices:
                if df.at[idx, 'Date'] == d_str:
                    home = df.at[idx, 'Home']
                    away = df.at[idx, 'Away']
                    pred_ia = df.at[idx, 'Predicted_Winner']
                    
                    real_winner = None
                    
                    # Qui a gagné ?
                    if team_results_day.get(home) == 'W': real_winner = home
                    elif team_results_day.get(away) == 'W': real_winner = away
                    
                    if real_winner:
                        df.at[idx, 'Real_Winner'] = real_winner
                        
                        # Check IA
                        df.at[idx, 'Result'] = "GAGNE" if pred_ia == real_winner else "PERDU"
                        
                        # Check User (si vote existant)
                        user_pred = df.at[idx, 'User_Prediction']
                        if pd.notna(user_pred) and user_pred != "":
                            df.at[idx, 'User_Result'] = "GAGNE" if user_pred == real_winner else "PERDU"
                            
                        updates += 1
                        print(f"      [MAJ] {home} vs {away} -> Vainqueur: {real_winner}")

            time.sleep(0.6) # Anti-ban

        except Exception as e:
            print(f"      [ERREUR] {e}")

    if updates > 0:
        df.to_csv(HISTORY_FILE, index=False)
        print(f"\n[SUCCES] {updates} résultats mis à jour dans le CSV.")
    else:
        print("\n[INFO] Rien à mettre à jour.")

if __name__ == "__main__":
    verify()