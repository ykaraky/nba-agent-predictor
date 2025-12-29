import pandas as pd
import os
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

HISTORY_FILE = 'data/bets_history.csv'
DATES_TO_RECOVER = ['2025-12-28', '2025-12-29']

def get_team_name(tid):
    try: return teams.find_team_name_by_id(tid)['full_name']
    except: return "Unknown"

def recover():
    print("--- RATTRAPAGE DES JOURS MANQUANTS ---")
    
    if not os.path.exists(HISTORY_FILE):
        print("❌ Pas de CSV.")
        return

    df = pd.read_csv(HISTORY_FILE)
    existing_combinations = set(zip(df['Date'], df['Home'], df['Away']))
    
    new_rows = []

    for d_str in DATES_TO_RECOVER:
        print(f"🔍 Scan du {d_str}...")
        try:
            board = scoreboardv2.ScoreboardV2(game_date=d_str)
            games = board.game_header.get_data_frame()
            
            # Filtre les vrais matchs
            games = games.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
            
            for _, game in games.iterrows():
                h_name = get_team_name(game['HOME_TEAM_ID'])
                a_name = get_team_name(game['VISITOR_TEAM_ID'])
                
                # Vérifie si déjà présent
                if (d_str, h_name, a_name) in existing_combinations:
                    print(f"   Déjà là : {h_name} vs {a_name}")
                    continue
                
                print(f"   ➕ AJOUT : {h_name} vs {a_name}")
                
                # On crée une ligne neutre (on ne refait pas la prédiction ML pour simplifier)
                new_row = {
                    "Date": d_str,
                    "Home": h_name,
                    "Away": a_name,
                    "Predicted_Winner": h_name, # Par défaut Domicile pour éviter vide
                    "Confidence": "50.0%",
                    "Type": "Auto",
                    "Result": None,
                    "Real_Winner": None,
                    "User_Prediction": None,
                    "User_Result": None,
                    "User_Reason": None
                }
                new_rows.append(new_row)
                
        except Exception as e:
            print(f"   Erreur API : {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Concaténer et sauvegarder
        df_final = pd.concat([df, new_df], ignore_index=True)
        df_final.to_csv(HISTORY_FILE, index=False)
        print(f"\n✅ {len(new_rows)} matchs ajoutés au CSV !")
    else:
        print("\nTout est déjà à jour.")

if __name__ == "__main__":
    recover()