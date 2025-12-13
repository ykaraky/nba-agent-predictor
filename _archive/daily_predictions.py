import pandas as pd
import xgboost as xgb
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams
from datetime import datetime

# --- CONFIGURATION ---
today = datetime.now().strftime('%Y-%m-%d')
print(f"--- PRÉDICTIONS POUR LE {today} ---")

# 1. Dictionnaire des équipes
nba_teams = teams.get_teams()
team_lookup = {team['id']: team['abbreviation'] for team in nba_teams}

# 2. Chargement du modèle
print("Chargement du cerveau et de l'historique...")
model = xgb.XGBClassifier()
try:
    model.load_model("nba_predictor.json")
    df_history = pd.read_csv('nba_games_ready.csv')
    df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
    df_history['TEAM_ID'] = df_history['TEAM_ID'].astype(int)
except Exception as e:
    print(f"Erreur de chargement : {e}")
    exit()

# 3. Récupération des matchs
print("Récupération des matchs du jour sur internet...")
try:
    board = scoreboardv2.ScoreboardV2(game_date=today)
    games_today = board.game_header.get_data_frame()
except Exception as e:
    print("Impossible de contacter l'API NBA.")
    exit()

# --- CORRECTION DU BUG ICI ---
# On affiche le nombre brut
count_raw = len(games_today)

# On supprime les lignes où les IDs des équipes sont vides (NaN ou None)
games_today = games_today.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])

# On supprime les lignes où le match n'a pas encore de statut clair (Optionnel mais plus sûr)
count_clean = len(games_today)

if count_clean == 0:
    print(f"Aucun match valide trouvé pour le {today} (Brut: {count_raw}).")
    exit()

print(f"\n{count_clean} matchs valides trouvés (Nettoyage effectué). Analyse en cours...\n")

# 4. Fonction stats
def get_team_features(team_id, game_date_str):
    try:
        team_id = int(team_id) # Sécurité
        team_history = df_history[df_history['TEAM_ID'] == team_id].sort_values('GAME_DATE')
        
        if len(team_history) == 0:
            return None
            
        last_game = team_history.iloc[-1]
        last_game_date = last_game['GAME_DATE']
        
        current_date = pd.to_datetime(game_date_str)
        days_rest = (current_date - last_game_date).days
        
        return {
            'PTS_LAST_5': last_game['PTS_LAST_5'],
            'WIN_LAST_5': last_game['WIN_LAST_5'],
            'DAYS_REST': min(days_rest, 7)
        }
    except Exception:
        return None

# 5. Boucle principale
for index, game in games_today.iterrows():
    try:
        # On sécurise la récupération des IDs
        home_id = int(game['HOME_TEAM_ID'])
        away_id = int(game['VISITOR_TEAM_ID'])
        
        home_name = team_lookup.get(home_id, str(home_id))
        away_name = team_lookup.get(away_id, str(away_id))
        
        match_label = f"{home_name} (Dom) vs {away_name} (Ext)"

        stats_home = get_team_features(home_id, today)
        stats_away = get_team_features(away_id, today)
        
        if stats_home and stats_away:
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
            
            print(f"🏀 {match_label}")
            
            # Affichage fatigue
            if stats_home['DAYS_REST'] <= 1: print(f"   ⚠️  Fatigue : {home_name} joue sans repos (B2B)")
            if stats_away['DAYS_REST'] <= 1: print(f"   ⚠️  Fatigue : {away_name} joue sans repos (B2B)")
            
            if prob_home > 0.5:
                conf = prob_home * 100
                print(f"   🏆 VAINQUEUR : {home_name} ({conf:.1f}%)")
            else:
                conf = (1 - prob_home) * 100
                print(f"   🏆 VAINQUEUR : {away_name} ({conf:.1f}%)")
            print("-" * 30)
            
        else:
            print(f"❌ {match_label} : Stats historiques introuvables.")
            
    except Exception as e:
        print(f"Erreur sur une ligne de match : {e}")
        continue

input("\nAppuie sur Entrée pour fermer...")