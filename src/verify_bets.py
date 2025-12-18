import pandas as pd
import os
import time
from nba_api.stats.static import teams

# --- CONFIG ---
HISTORY_FILE = 'data/bets_history.csv'
GAMES_FILE = 'data/nba_games.csv'

# --- OUTILS ---
def get_team_map():
    """Crée un dictionnaire robuste pour retrouver l'ID d'une équipe"""
    nba_teams = teams.get_teams()
    mapping = {}
    for t in nba_teams:
        # On map tout ce qu'on peut : Nom complet, Code, Surnom, Ville
        mapping[t['full_name'].upper()] = t['id']
        mapping[t['abbreviation'].upper()] = t['id']
        mapping[t['nickname'].upper()] = t['id']
        mapping[t['city'].upper()] = t['id']
        # Cas speciaux manuels si besoin
        mapping['NYK KNICKS'] = 1610612752
        mapping['SAS SPURS'] = 1610612759
    return mapping, nba_teams

def clean_team_name(name, mapping):
    """Tente de trouver l'ID d'une équipe depuis un nom approximatif"""
    if pd.isna(name): return None
    name = str(name).upper().strip()
    
    # 1. Correspondance directe
    if name in mapping: return mapping[name]
    
    # 2. Recherche mot-clé (ex: "Lakers" dans "LA Lakers")
    for key, tid in mapping.items():
        if key in name.split(): # Si "LAKERS" est un mot complet dans le nom
            return tid
            
    return None

def verify():
    print("--- VERIFICATION v6 (ID Based) ---")

    if not os.path.exists(HISTORY_FILE) or not os.path.exists(GAMES_FILE):
        print("[ERREUR] Fichiers manquants (history ou games).")
        return

    # 1. CHARGEMENT
    try:
        bets = pd.read_csv(HISTORY_FILE)
        games = pd.read_csv(GAMES_FILE)
        
        # Conversion Dates
        bets['Date'] = pd.to_datetime(bets['Date'], errors='coerce')
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'], errors='coerce')
        
    except Exception as e:
        print(f"[ERREUR] Lecture CSV: {e}")
        return

    # 2. MIGRATION V6 (Ajout des colonnes User si absentes)
    cols_added = False
    if 'User_Prediction' not in bets.columns:
        bets['User_Prediction'] = None
        cols_added = True
    if 'User_Result' not in bets.columns:
        bets['User_Result'] = None
        cols_added = True
        
    if cols_added:
        print("[INFO] Colonnes v6 ajoutees au CSV.")

    # 3. VERIFICATION
    team_map, team_list = get_team_map()
    id_to_name = {t['id']: t['full_name'] for t in team_list}
    
    updated_count = 0
    
    for idx, row in bets.iterrows():
        # On ne verifie que ce qui n'est pas fini
        if row['Result'] in ['GAGNE', 'PERDU'] and pd.notna(row['Real_Winner']):
            continue
            
        # Identifier le match via la date et les equipes
        game_date = row['Date']
        
        # On cherche l'ID du prono (IA)
        pred_name = row['Predicted_Winner']
        pred_id = clean_team_name(pred_name, team_map)
        
        if not pred_id:
            continue # Nom equipe inconnu

        # On cherche le match dans la base NBA (Meme jour ou +1 jour pour decalage horaire)
        # On cherche une ligne ou cette equipe a joue a cette date (approx)
        mask_date = (games['GAME_DATE'] >= game_date) & (games['GAME_DATE'] <= game_date + pd.Timedelta(days=1))
        match_row = games[mask_date & (games['TEAM_ID'] == pred_id)]
        
        if not match_row.empty:
            # Match trouve !
            game_data = match_row.iloc[0] # Prendre la premiere ligne trouvee
            
            # Resultat
            is_win = (game_data['WL'] == 'W')
            real_winner_name = id_to_name.get(pred_id) if is_win else "Autre"
            
            # Si l'IA a perdu, qui a gagne ? (On doit trouver l'adversaire)
            if not is_win:
                match_id = game_data['GAME_ID']
                # Trouver l'autre ligne du meme match
                opponent_row = games[(games['GAME_ID'] == match_id) & (games['TEAM_ID'] != pred_id)]
                if not opponent_row.empty:
                    opp_id = opponent_row.iloc[0]['TEAM_ID']
                    real_winner_name = id_to_name.get(opp_id, "Inconnu")
                else:
                    # Fallback si on a pas l'autre ligne (rare)
                    real_winner_name = row['Away'] if str(row['Predicted_Winner']) == row['Home'] else row['Home']

            # MISE A JOUR IA
            bets.at[idx, 'Result'] = "GAGNE" if is_win else "PERDU"
            bets.at[idx, 'Real_Winner'] = real_winner_name
            
            # MISE A JOUR USER (v6)
            user_pred = row.get('User_Prediction')
            if pd.notna(user_pred) and str(user_pred) != "":
                user_win = (clean_team_name(user_pred, team_map) == clean_team_name(real_winner_name, team_map))
                bets.at[idx, 'User_Result'] = "GAGNE" if user_win else "PERDU"

            updated_count += 1
            print(f"[MAJ] {game_date.date()} : {pred_name} -> {bets.at[idx, 'Result']}")

    # 4. SAUVEGARDE
    if updated_count > 0 or cols_added:
        # Reformater la date en string pour le CSV
        bets['Date'] = bets['Date'].dt.strftime('%Y-%m-%d')
        bets.to_csv(HISTORY_FILE, index=False)
        print(f"[OK] {updated_count} resultats mis a jour.")
    else:
        print("[INFO] Tout est a jour.")

if __name__ == "__main__":
    verify()