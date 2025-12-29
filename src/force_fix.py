import pandas as pd
import os
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

HISTORY_FILE = 'data/bets_history.csv'
# On cible uniquement le 28 car le 29 n'est pas joué
TARGET_DATES = ['2025-12-28'] 

def get_team_map():
    nba_teams = teams.get_teams()
    return {t['id']: t['full_name'] for t in nba_teams}

def normalize_date(d_str):
    """Essaie de convertir n'importe quelle date en YYYY-MM-DD"""
    try:
        # Tente YYYY-MM-DD
        return datetime.strptime(str(d_str).strip(), '%Y-%m-%d').strftime('%Y-%m-%d')
    except:
        try:
            # Tente DD.MM.YYYY (Format Suisse/Streamlit parfois)
            return datetime.strptime(str(d_str).strip(), '%d.%m.%Y').strftime('%Y-%m-%d')
        except:
            return str(d_str)

def force_fix():
    print(f"--- RÉPARATION ROBUSTE : {TARGET_DATES} ---")
    
    if not os.path.exists(HISTORY_FILE):
        print("❌ Pas de CSV.")
        return

    df = pd.read_csv(HISTORY_FILE)
    id_to_name = get_team_map()
    updates = 0

    for date_target in TARGET_DATES:
        print(f"\n📅 Cible : {date_target}")
        
        # 1. API
        d_us = datetime.strptime(date_target, '%Y-%m-%d').strftime('%m/%d/%Y')
        finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=d_us, date_to_nullable=d_us, league_id_nullable='00'
        )
        results = finder.get_data_frames()[0]
        
        if results.empty:
            print("   ⚠️ Pas de matchs API (Trop tôt ?).")
            continue
            
        # Dico résultats API
        day_results = {}
        for _, r in results.iterrows():
            tid = int(r['TEAM_ID'])
            if tid in id_to_name:
                name = id_to_name[tid]
                outcome = "GAGNE" if r['WL'] == 'W' else "PERDU"
                day_results[name.upper()] = outcome

        # 2. CSV MATCHING
        for idx, row in df.iterrows():
            # On normalise la date du CSV pour être sûr
            row_date_clean = normalize_date(row['Date'])
            
            if row_date_clean == date_target:
                home = str(row['Home']).strip()
                away = str(row['Away']).strip()
                
                # Check Resultats
                res_home = day_results.get(home.upper())
                res_away = day_results.get(away.upper())
                
                real_winner = None
                if res_home == 'GAGNE': real_winner = home
                elif res_away == 'GAGNE': real_winner = away
                
                print(f"   MATCH TROUVÉ CSV : {home} vs {away}")
                
                if real_winner:
                    df.at[idx, 'Real_Winner'] = real_winner
                    pred = str(row['Predicted_Winner']).strip()
                    df.at[idx, 'Result'] = "GAGNE" if pred == real_winner else "PERDU"
                    
                    user_pred = str(row['User_Prediction']).strip()
                    if user_pred and user_pred != "nan":
                        df.at[idx, 'User_Result'] = "GAGNE" if user_pred == real_winner else "PERDU"
                    
                    updates += 1
                    print(f"      ✅ WINNER UPDATE : {real_winner}")
                else:
                    print(f"      ⚠️ Pas de vainqueur détecté via API (Noms : {home}/{away})")

    if updates > 0:
        df.to_csv(HISTORY_FILE, index=False)
        print(f"\n💾 {updates} corrections appliquées !")
    else:
        print("\nZéro mise à jour. Vérifie les noms d'équipes ou les dates.")

if __name__ == "__main__":
    force_fix()