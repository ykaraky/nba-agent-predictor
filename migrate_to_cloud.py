import pandas as pd
import requests # On utilise requests au lieu de supabase
import json
import os
import numpy as np

# --- 1. CONFIGURATION (A REMPLACER) ---
# Ton URL ressemble à : https://xyzxyzxyz.supabase.co
SUPABASE_URL = "https://meqvmpqyizffzlvomqbb.supabase.co" 
# Ta clé 'anon' / 'public'
SUPABASE_KEY = "sb_publishable_bSPoeHBKUrxsEwn0ZI5cdA_iAAK3wza"

# --- 2. PREPARATION DE L'API ---
# L'adresse exacte de ta table pour l'API REST
endpoint = f"{SUPABASE_URL}/rest/v1/bets_history"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal" # Pour aller plus vite
}

# --- 3. LECTURE DU CSV LOCAL ---
csv_path = "data/bets_history.csv"
if not os.path.exists(csv_path):
    print("[ERREUR] Fichier CSV introuvable.")
    exit()

df = pd.read_csv(csv_path)
print(f"Lecture de {len(df)} lignes...")

# Nettoyage : Remplacer NaN (vide) par None (null pour JSON)
df = df.replace({np.nan: None})

# --- 4. CONVERSION ET ENVOI ---
rows_to_insert = []

for index, row in df.iterrows():
    # Construction de l'objet (Dictionnaire)
    data = {
        "game_date": row['Date'],
        "home_team": row['Home'],
        "away_team": row['Away'],
        "predicted_winner": row['Predicted_Winner'],
        "confidence": str(row['Confidence']),
        "result_ia": row['Result'],
        "real_winner": row['Real_Winner'],
        # Gestion safe des colonnes v6/v8
        "user_prediction": row.get('User_Prediction', None),
        "user_result": row.get('User_Result', None),
        "user_reason": row.get('User_Reason', None)
    }
    rows_to_insert.append(data)

print(f"Envoi de {len(rows_to_insert)} lignes vers le Cloud...")

try:
    # On envoie tout d'un coup (Batch)
    response = requests.post(endpoint, headers=headers, json=rows_to_insert)
    
    if response.status_code in [200, 201]:
        print("[SUCCES] Migration terminee ! Tes donnees sont dans Supabase.")
    else:
        print(f"[ERREUR] Code {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"[CRASH] {e}")