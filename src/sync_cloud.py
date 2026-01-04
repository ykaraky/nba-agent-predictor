import pandas as pd
import requests
import json
import os
import numpy as np
from dotenv import load_dotenv

# Chargement des secrets
load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CSV_PATH = "data/bets_history.csv"

def sync_to_supabase():
    print("--- SYNCHRONISATION VERS SUPABASE (CLEAN & DEDUP) ---")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("[ERREUR] Clés manquantes.")
        return

    if not os.path.exists(CSV_PATH):
        print(f"[ERREUR] CSV introuvable : {CSV_PATH}")
        return

    # 1. Lecture
    try:
        df = pd.read_csv(CSV_PATH)
        count_before = len(df)
        
        # 2. NETTOYAGE DOUBLONS (CRITIQUE)
        # On garde la dernière version de chaque match (Date + Home + Away)
        df = df.drop_duplicates(subset=['Date', 'Home', 'Away'], keep='last')
        count_after = len(df)
        
        # DELTA OPTIMIZATION: Only sync the last 100 items
        # This prevents sending 3000+ rows every morning while keeping latest results updated
        if len(df) > 100:
            print(f"[INFO] Delta Sync actif : Envoi des 100 derniers paris uniquement.")
            df = df.tail(100)
            
        if count_before > count_after:
            print(f"[INFO] {count_before - count_after} doublons supprimés du CSV avant envoi.")
            # Optionnel : Sauvegarder le CSV propre
            df.to_csv(CSV_PATH, index=False)

        df = df.replace({np.nan: None})
        
    except Exception as e:
        print(f"[ERREUR] Traitement CSV : {e}")
        return

    # 3. Préparation des données (Tout l'historique propre)
    rows = []
    for _, row in df.iterrows():
        row_data = {
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
            "type": row.get('Type', 'Auto')
        }
        rows.append(row_data)

    # 4. Envoi (Upsert)
    endpoint = f"{SUPABASE_URL}/rest/v1/bets_history?on_conflict=game_date,home_team,away_team"
    
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    try:
        print(f"[INFO] Envoi de {len(rows)} lignes uniques...")
        r = requests.post(endpoint, headers=headers, json=rows)
        
        if r.status_code in [200, 201]:
            print(f"[SUCCES] Supabase synchronisé ({len(rows)} matchs) !")
        else:
            print(f"[ERREUR API] {r.status_code} - {r.text}")
            
    except Exception as e:
        print(f"[CRASH] {e}")

if __name__ == "__main__":
    sync_to_supabase()