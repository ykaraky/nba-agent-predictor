import pandas as pd
import requests
import json
import os
import numpy as np
from dotenv import load_dotenv # <--- AJOUT

# Chargement des secrets depuis .env
load_dotenv()

# --- CONFIGURATION SECURISEE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CSV_PATH = "data/bets_history.csv"

def sync_to_supabase():
    print("--- SYNCHRONISATION CLOUD (V2) ---")
    
    if not os.path.exists(CSV_PATH):
        print("[SKIP] Pas de CSV trouvé.")
        return

    df = pd.read_csv(CSV_PATH)
    # Nettoyage : NaN devient None (null)
    df = df.replace({np.nan: None})
    
    # On prend les 100 derniers matchs pour être large et mettre à jour les types
    df_recent = df.tail(100) 

    rows = []
    for _, row in df_recent.iterrows():
        # Construction de l'objet JSON complet
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
            "user_reason": row.get('User_Reason'),
            "type": row.get('Type', 'Auto') # <--- AJOUT DE LA COLONNE TYPE
        })

    # Endpoint avec option UPSERT (Mise à jour si conflit sur la clé unique)
    endpoint = f"{SUPABASE_URL}/rest/v1/bets_history?on_conflict=game_date,home_team,away_team"
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    try:
        print(f"Envoi de {len(rows)} lignes...")
        r = requests.post(endpoint, headers=headers, json=rows)
        
        if r.status_code in [200, 201]:
            print(f"[OK] Synchronisation réussie (Types mis à jour).")
        else:
            print(f"[ERREUR] {r.status_code} - {r.text}")
            
    except Exception as e:
        print(f"[CRASH] {e}")

if __name__ == "__main__":
    sync_to_supabase()