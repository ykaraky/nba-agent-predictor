import pandas as pd
import xgboost as xgb
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

# --- CONFIGURATION ---
TARGET_DATE = datetime.now().strftime('%Y-%m-%d')
print(f"\n🏀 --- PRÉDICTEUR HYBRIDE (Date : {TARGET_DATE}) --- 🏀")

# 1. Chargement des données et du modèle
print("Chargement du cerveau et de l'historique...")
try:
    model = xgb.XGBClassifier()
    model.load_model("nba_predictor.json")
    
    df_history = pd.read_csv('nba_games_ready.csv')
    df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
    
    # Dictionnaire des équipes pour l'affichage auto
    nba_teams = teams.get_teams()
    team_lookup = {team['id']: team['abbreviation'] for team in nba_teams}
    
except Exception as e:
    print(f"❌ Erreur critique de chargement : {e}")
    exit()

# 2. Fonction Cœur : Récupère stats + Fatigue
def get_team_stats(team_abbr_or_id, target_date_str):
    # Si c'est un ID (mode auto), on le garde, sinon on cherche par abréviation (mode manuel)
    if str(team_abbr_or_id).isdigit():
        # Recherche par ID
        team_id = int(team_abbr_or_id)
        team_games = df_history[df_history['TEAM_ID'] == team_id].sort_values('GAME_DATE')
        team_name = team_lookup.get(team_id, str(team_id)) # Pour affichage debug
    else:
        # Recherche par Texte (LAL)
        team_abbr = str(team_abbr_or_id).upper()
        team_games = df_history[df_history['MATCHUP'].str.contains(team_abbr)].sort_values('GAME_DATE')
        team_name = team_abbr

    if len(team_games) == 0:
        return None
        
    last_game = team_games.iloc[-1]
    last_game_date = last_game['GAME_DATE']
    
    # Calcul des jours de repos
    target_date = pd.to_datetime(target_date_str)
    days_rest = (target_date - last_game_date).days
    
    return {
        'NAME': team_name,
        'PTS_LAST_5': last_game['PTS_LAST_5'],
        'WIN_LAST_5': last_game['WIN_LAST_5'],
        'DAYS_REST': min(days_rest, 7),
        'LAST_GAME_DATE': last_game_date
    }

def make_prediction(home_id_or_name, away_id_or_name):
    stats_home = get_team_stats(home_id_or_name, TARGET_DATE)
    stats_away = get_team_stats(away_id_or_name, TARGET_DATE)
    
    if stats_home and stats_away:
        home_name = stats_home['NAME']
        away_name = stats_away['NAME']
        
        print(f"\n📊 {home_name} (Dom) vs {away_name} (Ext)")
        
        # Alerte Fatigue
        if stats_home['DAYS_REST'] <= 1: print(f"  ⚠️ FATIGUE : {home_name} est en Back-to-back !")
        if stats_away['DAYS_REST'] <= 1: print(f"  ⚠️ FATIGUE : {away_name} est en Back-to-back !")
        
        # Création Dataframe
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
        
        if prob_home > 0.5:
            conf = prob_home * 100
            print(f"🏆 VAINQUEUR : {home_name} ({conf:.1f}%)")
            return home_name, prob_home
        else:
            conf = (1 - prob_home) * 100
            print(f"🏆 VAINQUEUR : {away_name} ({conf:.1f}%)")
            return away_name, (1 - prob_home)
    else:
        print("❌ Données historiques manquantes pour l'une des équipes.")
        return None, None

# --- PARTIE 1 : TENTATIVE AUTOMATIQUE ---
print("\n🔄 Tentative de récupération automatique des matchs du jour...")
try:
    board = scoreboardv2.ScoreboardV2(game_date=TARGET_DATE, timeout=5) # Timeout court
    games = board.game_header.get_data_frame()
    
    # Nettoyage des lignes vides (bug API)
    games = games.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
    
    if len(games) > 0:
        print(f"✅ {len(games)} matchs trouvés via l'API !\n")
        for _, game in games.iterrows():
            make_prediction(game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID'])
            print("-" * 20)
    else:
        print("⚠️ Aucun match trouvé automatiquement (ou bug API).")
        
except Exception as e:
    print(f"⚠️ Mode automatique indisponible ({e}).")
    print("Passage au mode manuel.")

# --- PARTIE 2 : MODE MANUEL (FALLBACK) ---
print("\n" + "="*50)
print("🖐️  MODE MANUEL ACTIVÉ")
print("Tape les équipes toi-même (ou 'exit' pour quitter).")

while True:
    print("\n--- Nouveau Pronostic ---")
    h = input("Équipe DOMICILE (ex: LAL) : ").strip().upper()
    if h == 'EXIT': break
    
    a = input("Équipe EXTÉRIEUR (ex: BOS) : ").strip().upper()
    if a == 'EXIT': break
    
    winner, conf = make_prediction(h, a)
    
    # Calculateur de cote optionnel
    if winner:
        try:
            user_input = input(f"Cote du bookmaker pour {winner} ? (Entrée pour passer) : ")
            if user_input.strip():
                odds = float(user_input.replace(',', '.'))
                prob_book = 1 / odds
                prob_ia = conf
                edge = prob_ia - prob_book
                
                print(f"  -> Avantage calculé : {edge*100:.1f}%")
                if edge > 0.05: print("  ✅ VALUE BET !")
                elif edge > 0: print("  👌 OK.")
                else: print("  ❌ Mauvais pari.")
        except:
            pass