import pandas as pd
import requests
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CSV_PATH = "data/bets_history.csv"

def pull_votes_from_cloud():
    print("--- RÉCUPÉRATION DES VOTES (CLOUD -> LOCAL) ---")
    
    # 1. Lire le CSV Local
    if not os.path.exists(CSV_PATH):
        print("[ERREUR] Pas de CSV local.")
        return
    
    df_local = pd.read_csv(CSV_PATH)
    print(f"[LOCAL] {len(df_local)} lignes chargées.")

    # 2. Lire Supabase
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
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

    # 3. Fusion Intelligente
    updates_count = 0
    
    # On parcourt les données Cloud pour mettre à jour le Local
    for row_cloud in cloud_data:
        c_date = row_cloud.get('game_date')
        c_home = row_cloud.get('home_team')
        c_vote = row_cloud.get('user_prediction')
        c_reason = row_cloud.get('user_reason')
        
        # Si pas de vote dans le cloud, on passe
        if not c_vote: continue
        
        # On cherche la ligne correspondante dans le CSV local
        # Masque : Date ET Home Team correspondent
        mask = (df_local['Date'] == c_date) & (df_local['Home'] == c_home)
        
        if mask.any():
            # On regarde si le local est vide
            idx = df_local[mask].index[0]
            local_vote = df_local.at[idx, 'User_Prediction']
            
            # Si Local vide OU différent du Cloud -> On prend le Cloud
            if pd.isna(local_vote) or local_vote == "" or local_vote != c_vote:
                df_local.at[idx, 'User_Prediction'] = c_vote
                df_local.at[idx, 'User_Reason'] = c_reason
                updates_count += 1
                print(f"   -> Vote récupéré : {c_home} vs ... : {c_vote}")

    # 4. Sauvegarde
    if updates_count > 0:
        df_local.to_csv(CSV_PATH, index=False)
        print(f"\n[SUCCÈS] {updates_count} votes ont été rapatriés dans le CSV local.")
    else:
        print("\n[INFO] Local déjà à jour avec le Cloud.")

if __name__ == "__main__":
    pull_votes_from_cloud()