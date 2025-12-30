import pandas as pd
import requests
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CSV_PATH = "data/bets_history.csv"

def normalize_date(val):
    if pd.isna(val): return ""
    return str(val).split('T')[0].strip()

def normalize_str(val):
    if pd.isna(val): return ""
    return str(val).strip()

def pull_votes_from_cloud():
    print("--- RÉCUPÉRATION (UPDATE & INSERT) CLOUD -> LOCAL ---")
    
    if not os.path.exists(CSV_PATH):
        print("[ERREUR] Pas de CSV local.")
        return
    
    df_local = pd.read_csv(CSV_PATH)
    print(f"[LOCAL] {len(df_local)} lignes.")

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    # On récupère TOUT l'historique du cloud pour être sûr de ne rien rater
    url = f"{SUPABASE_URL}/rest/v1/bets_history?select=*"
    
    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            print(f"[ERREUR CLOUD] {r.text}")
            return
        cloud_data = r.json()
        print(f"[CLOUD] {len(cloud_data)} lignes récupérées.")
    except Exception as e:
        print(f"[CRASH] {e}")
        return

    # Préparation matching
    df_local['match_date_clean'] = df_local['Date'].apply(normalize_date)
    df_local['match_home_clean'] = df_local['Home'].apply(normalize_str)

    updates_count = 0
    new_rows = []
    
    for row_cloud in cloud_data:
        c_date = normalize_date(row_cloud.get('game_date'))
        c_home = normalize_str(row_cloud.get('home_team'))
        
        # Masque de recherche
        mask = (df_local['match_date_clean'] == c_date) & (df_local['match_home_clean'] == c_home)
        
        if mask.any():
            # --- CAS 1 : MISE A JOUR (Le match existe en local) ---
            idx = df_local[mask].index[0]
            
            # On met à jour les votes si nécessaire
            c_vote = row_cloud.get('user_prediction')
            c_reason = row_cloud.get('user_reason')
            l_vote = df_local.at[idx, 'User_Prediction']
            
            if c_vote and (pd.isna(l_vote) or l_vote != c_vote):
                df_local.at[idx, 'User_Prediction'] = c_vote
                df_local.at[idx, 'User_Reason'] = c_reason
                updates_count += 1
                print(f"   [MAJ] Vote récupéré : {c_home} ({c_date})")
                
        else:
            # --- CAS 2 : INSERTION (Le match manque en local) ---
            # C'est ce qui manquait pour le 30.12 !
            print(f"   [NOUVEAU] Import du match : {c_home} vs {row_cloud.get('away_team')} ({c_date})")
            
            new_row = {
                "Date": row_cloud.get('game_date'),
                "Home": row_cloud.get('home_team'),
                "Away": row_cloud.get('away_team'),
                "Predicted_Winner": row_cloud.get('predicted_winner'),
                "Confidence": row_cloud.get('confidence'),
                "Type": row_cloud.get('type', 'Auto'),
                "Result": row_cloud.get('result_ia'),
                "Real_Winner": row_cloud.get('real_winner'),
                "User_Prediction": row_cloud.get('user_prediction'),
                "User_Result": row_cloud.get('user_result'),
                "User_Reason": row_cloud.get('user_reason')
            }
            new_rows.append(new_row)

    # Nettoyage colonnes temp
    df_local = df_local.drop(columns=['match_date_clean', 'match_home_clean'])
    
    # Fusion
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_local = pd.concat([df_local, df_new], ignore_index=True)
        # On trie par date pour faire propre
        df_local['Date'] = pd.to_datetime(df_local['Date'])
        df_local = df_local.sort_values('Date')
        # Remise en string YYYY-MM-DD
        df_local['Date'] = df_local['Date'].dt.strftime('%Y-%m-%d')
        
    # Sauvegarde
    if updates_count > 0 or len(new_rows) > 0:
        df_local.to_csv(CSV_PATH, index=False)
        print(f"\n[SUCCÈS] {updates_count} mises à jour et {len(new_rows)} ajouts sauvegardés.")
    else:
        print("\n[INFO] Tout est déjà synchro.")

if __name__ == "__main__":
    pull_votes_from_cloud()