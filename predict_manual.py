import pandas as pd
import xgboost as xgb
from datetime import datetime

# --- CONFIGURATION ---
# On fixe la date d'aujourd'hui (ou celle du match que tu veux prédire)
# Tu peux changer cette date si tu veux prédire pour demain
TARGET_DATE = datetime.now().strftime('%Y-%m-%d')

print(f"--- PRÉDICTEUR MANUEL (Date cible : {TARGET_DATE}) ---")

# 1. Chargement des données
print("Chargement du cerveau...")
try:
    model = xgb.XGBClassifier()
    model.load_model("nba_predictor.json")
    
    df_history = pd.read_csv('nba_games_ready.csv')
    df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
    # Conversion des IDs en texte pour faciliter la recherche si besoin, 
    # mais ici on cherche par nom d'équipe dans le MATCHUP
except Exception as e:
    print(f"Erreur critique : {e}")
    exit()

# 2. Fonction intelligente
def get_team_stats(team_abbr, target_date_str):
    team_abbr = team_abbr.upper()
    
    # On cherche les matchs où l'équipe apparaît (ex: "LAL")
    team_games = df_history[df_history['MATCHUP'].str.contains(team_abbr)].sort_values('GAME_DATE')
    
    if len(team_games) == 0:
        return None
        
    last_game = team_games.iloc[-1]
    last_game_date = last_game['GAME_DATE']
    
    # Calcul de la fatigue par rapport à la date cible
    target_date = pd.to_datetime(target_date_str)
    days_rest = (target_date - last_game_date).days
    
    return {
        'PTS_LAST_5': last_game['PTS_LAST_5'],
        'WIN_LAST_5': last_game['WIN_LAST_5'],
        'DAYS_REST': min(days_rest, 7), # Plafond à 7
        'LAST_GAME_DATE': last_game_date
    }

# --- BOUCLE D'INTERACTION ---
while True:
    print("\n" + "="*40)
    print("NOUVEAU PRONOSTIC (Tape 'exit' pour quitter)")
    home_team = input("Équipe DOMICILE (ex: LAL) : ").strip()
    if home_team.lower() == 'exit': break
    
    away_team = input("Équipe EXTÉRIEUR (ex: BOS) : ").strip()
    if away_team.lower() == 'exit': break
    
    # Récupération des stats
    stats_home = get_team_stats(home_team, TARGET_DATE)
    stats_away = get_team_stats(away_team, TARGET_DATE)
    
    if stats_home and stats_away:
        print(f"\nAnalyse : {home_team} vs {away_team}")
        
        # Affichage Fatigue
        rest_home = stats_home['DAYS_REST']
        rest_away = stats_away['DAYS_REST']
        
        print(f"- Repos {home_team} : {rest_home} jours (Dernier match : {stats_home['LAST_GAME_DATE'].date()})")
        print(f"- Repos {away_team} : {rest_away} jours (Dernier match : {stats_away['LAST_GAME_DATE'].date()})")
        
        if rest_home <= 1: print(f"  ⚠️ ALERTE : {home_team} est fatigué (Back-to-back) !")
        if rest_away <= 1: print(f"  ⚠️ ALERTE : {away_team} est fatigué (Back-to-back) !")

        # Préparation pour l'IA
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
        
        print("\n--- RÉSULTAT ---")
        if prob_home > 0.5:
            conf = prob_home * 100
            print(f"🏆 Vainqueur : {home_team.upper()} ({conf:.1f}%)")
        else:
            conf = (1 - prob_home) * 100
            print(f"🏆 Vainqueur : {away_team.upper()} ({conf:.1f}%)")
            
    else:
        print("\n❌ Erreur : Une des équipes est introuvable dans l'historique.")
        print("Vérifie les abréviations (ex: UTILISE 'GSW' et pas 'WARRIORS')")