import pandas as pd
import xgboost as xgb
from datetime import datetime
import os
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

print("--- GÉNÉRATION AUTOMATIQUE DES PRONOSTICS ---")

# 1. Chargement des ressources
try:
    if os.path.exists("nba_predictor.json"):
        model = xgb.XGBClassifier()
        model.load_model("nba_predictor.json")
    else:
        print("❌ Erreur : nba_predictor.json introuvable.")
        exit()

    if os.path.exists('nba_games_ready.csv'):
        df_history = pd.read_csv('nba_games_ready.csv')
        df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
    else:
        print("❌ Erreur : nba_games_ready.csv introuvable.")
        exit()
        
    nba_teams = teams.get_teams()
    # Dico pour avoir les noms propres
    id_to_name = {t['id']: f"{t['abbreviation']} {t['nickname']}" for t in nba_teams}

except Exception as e:
    print(f"❌ Erreur chargement : {e}")
    exit()

# 2. Fonction de Prédiction (Copie de la logique de l'app)
def get_prediction_logic(home_id, away_id):
    home_games = df_history[df_history['TEAM_ID'] == home_id].sort_values('GAME_DATE')
    away_games = df_history[df_history['TEAM_ID'] == away_id].sort_values('GAME_DATE')
    
    if home_games.empty or away_games.empty: return None

    last_home = home_games.iloc[-1]
    last_away = away_games.iloc[-1]
    
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    rest_home = (today - last_home['GAME_DATE']).days
    rest_away = (today - last_away['GAME_DATE']).days
    
    input_data = pd.DataFrame([{
        'EFG_PCT_LAST_5_HOME': last_home['EFG_PCT_LAST_5'],
        'EFG_PCT_LAST_5_AWAY': last_away['EFG_PCT_LAST_5'],
        'TOV_PCT_LAST_5_HOME': last_home['TOV_PCT_LAST_5'],
        'TOV_PCT_LAST_5_AWAY': last_away['TOV_PCT_LAST_5'],
        'ORB_RAW_LAST_5_HOME': last_home['ORB_RAW_LAST_5'],
        'ORB_RAW_LAST_5_AWAY': last_away['ORB_RAW_LAST_5'],
        'DIFF_EFG': last_home['EFG_PCT_LAST_5'] - last_away['EFG_PCT_LAST_5'],
        'DIFF_TOV': last_home['TOV_PCT_LAST_5'] - last_away['TOV_PCT_LAST_5'],
        'DIFF_ORB': last_home['ORB_RAW_LAST_5'] - last_away['ORB_RAW_LAST_5'],
        'DIFF_WIN': last_home['WIN_LAST_5'] - last_away['WIN_LAST_5'],
        'DIFF_REST': min(rest_home, 7) - min(rest_away, 7)
    }])

    probs = model.predict_proba(input_data)[0]
    return probs[1] # Probabilité victoire domicile

# 3. Récupération des matchs du jour
try:
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"📅 Recherche des matchs pour le {today_str}...")
    
    board = scoreboardv2.ScoreboardV2(game_date=today_str)
    games = board.game_header.get_data_frame()
    games = games.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
    
    if games.empty:
        print("⚠️ Aucun match trouvé pour ce soir.")
        exit()
        
    print(f"✅ {len(games)} matchs trouvés.")
    
    # 4. Boucle de prédiction et sauvegarde
    new_bets = 0
    
    # Vérification fichier historique
    if not os.path.exists('bets_history.csv'):
        with open('bets_history.csv', 'w') as f:
            f.write("Date,Home,Away,Predicted_Winner,Confidence,Type,Result\n")
            
    # Chargement pour éviter doublons
    try:
        current_hist = pd.read_csv('bets_history.csv')
    except:
        current_hist = pd.DataFrame()

    for _, game in games.iterrows():
        h_id, a_id = game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']
        h_name = id_to_name.get(h_id, str(h_id))
        a_name = id_to_name.get(a_id, str(a_id))
        
        # Vérification doublon avant calcul
        already_exists = False
        if not current_hist.empty:
            match_exists = current_hist[
                (current_hist['Date'] == today_str) & 
                (current_hist['Home'] == h_name) & 
                (current_hist['Away'] == a_name)
            ]
            if not match_exists.empty:
                already_exists = True
        
        if not already_exists:
            prob_home = get_prediction_logic(h_id, a_id)
            
            if prob_home is not None:
                if prob_home > 0.5:
                    winner, conf = h_name, prob_home * 100
                else:
                    winner, conf = a_name, (1 - prob_home) * 100
                
                # Écriture
                with open('bets_history.csv', 'a') as f:
                    f.write(f"\n{today_str},{h_name},{a_name},{winner},{conf:.1f}%,Auto,")
                
                print(f"   -> {h_name} vs {a_name} : {winner} ({conf:.1f}%) [SAUVEGARDÉ]")
                new_bets += 1
        else:
            print(f"   -> {h_name} vs {a_name} : Déjà fait.")

    print(f"\nTerminé ! {new_bets} nouveaux pronostics ajoutés.")

except Exception as e:
    print(f"❌ Erreur globale : {e}")