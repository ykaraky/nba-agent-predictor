import pandas as pd
import requests
import json
import os
import numpy as np

# --- CONFIG ---
SUPABASE_URL = "https://meqvmpqyizffzlvomqbb.supabase.co"
SUPABASE_KEY = "sb_publishable_bSPoeHBKUrxsEwn0ZI5cdA_iAAK3wza"
CSV_PATH = "data/bets_history.csv"

def sync_to_supabase():
    print("--- SYNCHRONISATION CLOUD ---")
    
    if not os.path.exists(CSV_PATH):
        print("[SKIP] Pas de CSV trouvé.")
        return

    df = pd.read_csv(CSV_PATH)
    # Nettoyage : NaN devient None (null)
    df = df.replace({np.nan: None})
    
    # On prend les 50 derniers matchs pour la mise à jour rapide
    df_recent = df.tail(50) 

    rows = []
    for _, row in df_recent.iterrows():
        rows.append({
            "game_date": row['Date'],
            "home_team": row['Home'],
            "away_team": row['Away'],
            "predicted_winner": row['Predicted_Winner'],
            "confidence": str(row['Confidence']),
            "result_ia": row['Result'],
            "real_winner": row['Real_Winner'],
            "user_prediction": row.get('User_Prediction'),
            "user_result": row.get('User_Result'),
            "user_reason": row.get('User_Reason')
        })

    # --- MODIFICATION ICI : On précise les colonnes de conflit dans l'URL ---
    # Cela dit à Supabase : "Vérifie si ce trio existe déjà"
    endpoint = f"{SUPABASE_URL}/rest/v1/bets_history?on_conflict=game_date,home_team,away_team"
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates" # Force la mise à jour (Upsert)
    }

    try:
        r = requests.post(endpoint, headers=headers, json=rows)
        
        # 200 = OK, 201 = Created
        if r.status_code in [200, 201]:
            print(f"[OK] {len(rows)} matchs synchronisés (Upsert).")
        else:
            print(f"[ERREUR] {r.status_code} - {r.text}")
            
    except Exception as e:
        print(f"[CRASH] {e}")

if __name__ == "__main__":
    sync_to_supabase()