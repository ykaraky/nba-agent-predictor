import pandas as pd
import xgboost as xgb
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams
import os
import csv

# --- CONFIGURATION ---
TARGET_DATE = datetime.now().strftime('%Y-%m-%d')
HISTORY_FILE = 'bets_history.csv'

print(f"\n🏀 --- PRÉDICTEUR HYBRIDE (Date : {TARGET_DATE}) --- 🏀")

# 1. Chargement
print("Chargement du cerveau et de l'historique...")
try:
    model = xgb.XGBClassifier()
    model.load_model("nba_predictor.json")
    
    df_history = pd.read_csv('nba_games_ready.csv')
    df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
    
    nba_teams = teams.get_teams()
    team_lookup = {team['id']: team['abbreviation'] for team in nba_teams}
    
except Exception as e:
    print(f"❌ Erreur critique de chargement : {e}")
    exit()

# 2. Fonction de Sauvegarde (NOUVEAU)
def save_to_history(date, home, away, winner, conf, model_type="Auto"):
    # Vérifier si le fichier existe, sinon créer l'en-tête
    file_exists = os.path.isfile(HISTORY_FILE)
    
    with open(HISTORY_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Date', 'Home', 'Away', 'Predicted_Winner', 'Confidence', 'Type', 'Result'])
        
        # On écrit la ligne (Result est vide pour l'instant)
        writer.writerow([date, home, away, winner, f"{conf:.1f}%", model_type, ''])
    
    print(f"   💾 Pronostic enregistré dans {HISTORY_FILE}")

# 3. Fonction Stats
def get_team_stats(team_abbr_or_id, target_date_str):
    if str(team_abbr_or_id).isdigit():
        team_id = int(team_abbr_or_id)
        team_games = df_history[df_history['TEAM_ID'] == team_id].sort_values('GAME_DATE')
        team_name = team_lookup.get(team_id, str(team_id))
    else:
        team_abbr = str(team_abbr_or_id).upper()
        team_games = df_history[df_history['MATCHUP'].str.contains(team_abbr)].sort_values('GAME_DATE')
        team_name = team_abbr

    if len(team_games) == 0:
        return None
        
    last_game = team_games.iloc[-1]
    last_game_date = last_game['GAME_DATE']
    
    target_date = pd.to_datetime(target_date_str)
    days_rest = (target_date - last_game_date).days
    
    return {
        'NAME': team_name,
        'PTS_LAST_5': last_game['PTS_LAST_5'],
        'WIN_LAST_5': last_game['WIN_LAST_5'],
        'DAYS_REST': min(days_rest, 7),
        'LAST_GAME_DATE': last_game_date
    }

def make_prediction(home_id_or_name, away_id_or_name, mode="Auto"):
    stats_home = get_team_stats(home_id_or_name, TARGET_DATE)
    stats_away = get_team_stats(away_id_or_name, TARGET_DATE)
    
    if stats_home and stats_away:
        home_name = stats_home['NAME']
        away_name = stats_away['NAME']
        
        print(f"\n📊 {home_name} (Dom) vs {away_name} (Ext)")
        
        if stats_home['DAYS_REST'] <= 1: print(f"  ⚠️ FATIGUE : {home_name} est en Back-to-back !")
        if stats_away['DAYS_REST'] <= 1: print(f"  ⚠️ FATIGUE : {away_name} est en Back-to-back !")
        
        input_data = pd.DataFrame([{
            'PTS_LAST_5_HOME': stats_home['PTS_LAST_5'],
            'PTS_LAST_5_AWAY': stats_away['PTS_LAST_5'],
            'WIN_LAST_5_HOME': stats_home['WIN_LAST_5'],
            'WIN_LAST_5_AWAY': stats_away['WIN_LAST_5'],
            'DAYS_REST_HOME': stats_home['DAYS_REST'],
            'DAYS_REST_AWAY': stats_away['DAYS_REST'],
            'DIFF_PTS': stats_home['PTS_LAST_5'] - stats_away['PTS_LAST_5'],
            'DIFF_REST': stats_home['DAYS_REST'] - stats_away['DAYS_REST']
        }])
        
        probs = model.predict_proba(input_data)[0]
        prob_home = probs[1]
        
        if prob_home > 0.5:
            conf = prob_home * 100
            winner = home_name
            print(f"🏆 VAINQUEUR : {home_name} ({conf:.1f}%)")
        else:
            conf = (1 - prob_home) * 100
            winner = away_name
            print(f"🏆 VAINQUEUR : {away_name} ({conf:.1f}%)")
            
        # Appel de la sauvegarde
        save_to_history(TARGET_DATE, home_name, away_name, winner, conf, mode)
        
        return winner, conf
    else:
        print("❌ Données historiques manquantes.")
        return None, None

# --- BOUCLES D'EXÉCUTION ---

print("\n🔄 Tentative automatique...")
try:
    board = scoreboardv2.ScoreboardV2(game_date=TARGET_DATE, timeout=5)
    games = board.game_header.get_data_frame()
    games = games.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
    
    if len(games) > 0:
        print(f"✅ {len(games)} matchs trouvés via l'API !\n")
        for _, game in games.iterrows():
            make_prediction(game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID'], "Auto")
            print("-" * 20)
    else:
        print("⚠️ Aucun match trouvé automatiquement.")
except Exception:
    print("⚠️ Mode automatique indisponible.")

print("\n" + "="*50)
print("🖐️  MODE MANUEL")
while True:
    h = input("\nÉquipe DOMICILE (ou 'exit') : ").strip().upper()
    if h == 'EXIT': break
    a = input("Équipe EXTÉRIEUR : ").strip().upper()
    if a == 'EXIT': break
    
    make_prediction(h, a, "Manuel")