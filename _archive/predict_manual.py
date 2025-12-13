import pandas as pd
import xgboost as xgb
from datetime import datetime

# --- CONFIGURATION ---
# Date du jour pour le calcul de la fatigue
TARGET_DATE = datetime.now().strftime('%Y-%m-%d')

print(f"--- PRÉDICTEUR MANUEL (Date référence : {TARGET_DATE}) ---")

# 1. Chargement des données et du modèle
print("Chargement du cerveau et de l'historique...")
try:
    model = xgb.XGBClassifier()
    model.load_model("nba_predictor.json")
    
    df_history = pd.read_csv('nba_games_ready.csv')
    df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
    
except Exception as e:
    print(f"Erreur critique de chargement : {e}")
    print("Vérifie que 'nba_predictor.json' et 'nba_games_ready.csv' sont bien dans le dossier.")
    exit()

# 2. Fonction pour récupérer les stats et calculer la fatigue
def get_team_stats(team_abbr, target_date_str):
    team_abbr = team_abbr.upper()
    
    # Recherche de l'équipe dans l'historique (colonne MATCHUP)
    team_games = df_history[df_history['MATCHUP'].str.contains(team_abbr)].sort_values('GAME_DATE')
    
    if len(team_games) == 0:
        return None
        
    last_game = team_games.iloc[-1]
    last_game_date = last_game['GAME_DATE']
    
    # Calcul des jours de repos
    target_date = pd.to_datetime(target_date_str)
    days_rest = (target_date - last_game_date).days
    
    return {
        'PTS_LAST_5': last_game['PTS_LAST_5'],
        'WIN_LAST_5': last_game['WIN_LAST_5'],
        'DAYS_REST': min(days_rest, 7), # On plafonne à 7 jours max
        'LAST_GAME_DATE': last_game_date
    }

# --- BOUCLE PRINCIPALE ---
while True:
    print("\n" + "="*50)
    print("NOUVEAU PRONOSTIC (Tape 'exit' pour quitter)")
    
    home_team = input("Équipe DOMICILE (ex: LAL) : ").strip().upper()
    if home_team == 'EXIT': break
    
    away_team = input("Équipe EXTÉRIEUR (ex: BOS) : ").strip().upper()
    if away_team == 'EXIT': break
    
    # Récupération des stats
    stats_home = get_team_stats(home_team, TARGET_DATE)
    stats_away = get_team_stats(away_team, TARGET_DATE)
    
    if stats_home and stats_away:
        print(f"\n📊 Analyse : {home_team} vs {away_team}")
        
        # Affichage Infos Fatigue
        rest_home = stats_home['DAYS_REST']
        rest_away = stats_away['DAYS_REST']
        print(f"- Repos {home_team} : {rest_home} jours (Dernier match : {stats_home['LAST_GAME_DATE'].date()})")
        print(f"- Repos {away_team} : {rest_away} jours (Dernier match : {stats_away['LAST_GAME_DATE'].date()})")
        
        if rest_home <= 1: print(f"  ⚠️ ALERTE FATIGUE : {home_team} est en Back-to-back !")
        if rest_away <= 1: print(f"  ⚠️ ALERTE FATIGUE : {away_team} est en Back-to-back !")

        # Préparation des données pour l'IA
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
        
        # Prédiction
        probs = model.predict_proba(input_data)[0]
        prob_home = probs[1]
        
        # Affichage du Vainqueur
        print("\n--- PRÉDICTION IA ---")
        if prob_home > 0.5:
            conf = prob_home * 100
            print(f"🏆 VAINQUEUR : {home_team} ({conf:.1f}%)")
        else:
            conf = (1 - prob_home) * 100
            print(f"🏆 VAINQUEUR : {away_team} ({conf:.1f}%)")

    else:
        print("\n❌ Erreur : Une des équipes est introuvable.")
        print("Vérifie les abréviations (ex: utilise 'GSW' et pas 'WARRIORS').")