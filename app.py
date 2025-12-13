import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime
import os
import sys
import subprocess
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2

# --- CONFIGURATION ---
st.set_page_config(page_title="NBA Manager v3.1", page_icon="🏀", layout="wide")

# --- MEMOIRE DE SESSION (PERSISTANCE) ---
# C'est ici qu'on stocke les résultats pour qu'ils ne disparaissent pas
if 'games_today' not in st.session_state:
    st.session_state['games_today'] = None
if 'last_run_date' not in st.session_state:
    st.session_state['last_run_date'] = None

# --- FONCTIONS ---

@st.cache_resource
def load_model_resource():
    model = xgb.XGBClassifier()
    try:
        if os.path.exists("nba_predictor.json"):
            model.load_model("nba_predictor.json")
            return model
        return None
    except:
        return None

@st.cache_data
def load_data_resource():
    try:
        if os.path.exists('nba_games_ready.csv'):
            df = pd.read_csv('nba_games_ready.csv')
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            return df
        return None
    except:
        return None

def get_team_list():
    nba_teams = teams.get_teams()
    return {f"{t['abbreviation']} - {t['nickname']}": t['id'] for t in nba_teams}

def get_last_modified_time(filename):
    """Récupère la date de modification d'un fichier pour savoir quand on l'a mis à jour"""
    if os.path.exists(filename):
        timestamp = os.path.getmtime(filename)
        return datetime.fromtimestamp(timestamp).strftime('%d/%m/%Y à %H:%M')
    return "Jamais"

def run_script_step(script_name, step_name, status_container):
    try:
        status_container.write(f"⏳ {step_name}...")
        subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        status_container.error(f"❌ Erreur dans {step_name}")
        st.code(e.stderr if e.stderr else str(e))
        return False

def get_prediction(model, df_history, home_id, away_id):
    home_games = df_history[df_history['TEAM_ID'] == home_id].sort_values('GAME_DATE')
    away_games = df_history[df_history['TEAM_ID'] == away_id].sort_values('GAME_DATE')
    
    if home_games.empty or away_games.empty: return None, None

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
    return probs[1], {'rh': min(rest_home,7), 'ra': min(rest_away,7)}

def save_bet(home, away, winner, conf, type_bet):
    if not os.path.exists('bets_history.csv'):
        with open('bets_history.csv', 'w') as f:
            f.write("Date,Home,Away,Predicted_Winner,Confidence,Type,Result\n")
    
    # On vérifie les doublons AVANT d'écrire pour éviter de spammer le fichier
    try:
        current_df = pd.read_csv('bets_history.csv')
        today = datetime.now().strftime('%Y-%m-%d')
        # Si une ligne existe déjà avec la même date, home et away, on n'écrit pas
        exists = not current_df[(current_df['Date'] == today) & (current_df['Home'] == home) & (current_df['Away'] == away)].empty
        if exists: return # On quitte silencieusement
    except:
        pass

    with open('bets_history.csv', 'a') as f:
        date = datetime.now().strftime('%Y-%m-%d')
        f.write(f"\n{date},{home},{away},{winner},{conf:.1f}%,{type_bet},")

# --- INTERFACE ---

st.title("🏀 NBA Manager v3.1")

model = load_model_resource()
df = load_data_resource()
teams_dict = get_team_list()
id_to_name = {v: k.split(' - ')[0] for k, v in teams_dict.items()}

tab1, tab2, tab3, tab4 = st.tabs(["🌞 Matchs du Jour", "🔮 Manuel", "📊 Bilan", "⚙️ Maintenance"])

# --- TAB 1 : AUTO PREDICT (Avec Mémoire) ---
with tab1:
    st.header("Routine Matinale")
    
    # Affichage du dernier run si existant
    if st.session_state['last_run_date']:
        st.caption(f"Dernière analyse : {st.session_state['last_run_date']}")

    if st.button("🚀 LANCER LA JOURNEE", type="primary"):
        status_box = st.status("Traitement en cours...", expanded=True)
        
        # 1. Pipeline de mise à jour
        if run_script_step('data_nba.py', "Mise a jour Donnees", status_box):
            if run_script_step('features_nba.py', "Calcul Stats", status_box):
                run_script_step('verify_bets.py', "Verification Paris", status_box)
                
                load_data_resource.clear()
                df = load_data_resource()
                
                status_box.write("🔎 Recherche matchs du soir...")
                
                # 2. Récupération Matchs
                try:
                    board = scoreboardv2.ScoreboardV2(game_date=datetime.now().strftime('%Y-%m-%d'))
                    games_raw = board.game_header.get_data_frame()
                    games_clean = games_raw.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
                    
                    # 3. Stockage en Session (C'est ça qui rend persistant !)
                    st.session_state['games_today'] = games_clean
                    st.session_state['last_run_date'] = datetime.now().strftime('%H:%M:%S')
                    
                    status_box.update(label="Terminé !", state="complete", expanded=False) # On ferme la boite
                    
                except Exception as e:
                    st.error(f"Erreur API : {e}")

    st.divider()

    # AFFICHAGE PERSISTANT
    # On vérifie si on a des matchs en mémoire
    if st.session_state['games_today'] is not None:
        games_df = st.session_state['games_today']
        
        if not games_df.empty:
            st.success(f"✅ {len(games_df)} matchs pour ce soir")
            
            for _, game in games_df.iterrows():
                h_id, a_id = game['HOME_TEAM_ID'], game['VISITOR_TEAM_ID']
                h_name = id_to_name.get(h_id, str(h_id))
                a_name = id_to_name.get(a_id, str(a_id))
                
                prob, det = get_prediction(model, df, h_id, a_id)
                
                if prob is not None:
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1: 
                        st.write(f"**{h_name}**")
                        if det['rh']<=1: st.error("Fatigue (B2B)")
                    with c3: 
                        st.write(f"**{a_name}**")
                        if det['ra']<=1: st.error("Fatigue (B2B)")
                    with c2:
                        if prob > 0.5:
                            win, conf, col = h_name, prob*100, "green"
                        else:
                            win, conf, col = a_name, (1-prob)*100, "red"
                        
                        st.markdown(f"<h3 style='text-align: center; color: {col}'>{win}</h3>", unsafe_allow_html=True)
                        st.progress(int(conf), f"Confiance: {conf:.1f}%")
                        
                        # Sauvegarde Auto sécurisée (anti-doublon)
                        save_bet(h_name, a_name, win, conf, "Auto")
                st.divider()
        else:
            st.info("Aucun match trouvé pour ce soir.")
    else:
        st.write("En attente de lancement...")

# --- TAB 2 : MANUEL (Vide par défaut) ---
with tab2:
    if model is None or df is None:
        st.warning("Donnees manquantes.")
    else:
        c1, c2 = st.columns(2)
        # index=None force l'utilisateur à choisir
        with c1: h_choice = st.selectbox("Domicile", list(teams_dict.keys()), index=None, placeholder="Choisis l'équipe...")
        with c2: a_choice = st.selectbox("Extérieur", list(teams_dict.keys()), index=None, placeholder="Choisis l'équipe...")
            
        # Le bouton n'apparait que si les équipes sont choisies
        if h_choice and a_choice:
            if st.button("Analyser le Duel"):
                prob, d = get_prediction(model, df, teams_dict[h_choice], teams_dict[a_choice])
                if prob is not None:
                    if prob > 0.5:
                        win, conf = h_choice.split(' - ')[0], prob*100
                        st.success(f"🏆 {win} ({conf:.1f}%)")
                    else:
                        win, conf = a_choice.split(' - ')[0], (1-prob)*100
                        st.success(f"🏆 {win} ({conf:.1f}%)")
                    save_bet(h_choice.split(' - ')[0], a_choice.split(' - ')[0], win, conf, "Manual")

# --- TAB 3 : BILAN (Nettoyage Amélioré) ---
with tab3:
    st.header("Historique")
    if os.path.exists('bets_history.csv'):
        hist = pd.read_csv('bets_history.csv')
        
        # 1. Gros Bouton pour nettoyer les doublons massifs
        col_clean, col_kpi = st.columns([1, 3])
        with col_clean:
            if st.button("🧹 Nettoyer tous les doublons"):
                before = len(hist)
                hist = hist.drop_duplicates(subset=['Date', 'Home', 'Away'], keep='last')
                hist.to_csv('bets_history.csv', index=False)
                after = len(hist)
                st.toast(f"{before - after} doublons supprimés !", icon="✨")
                time.sleep(1)
                st.rerun()

        st.dataframe(hist.sort_index(ascending=False))
        
        st.divider()
        st.subheader("🗑️ Suppression sélective")
        
        # Multiselect pour en supprimer plusieurs d'un coup
        rows_to_del = st.multiselect(
            "Sélectionner les lignes à supprimer (par Index)",
            options=hist.index.tolist(),
            placeholder="Ex: 0, 1, 5..."
        )
        
        if rows_to_del:
            if st.button(f"Supprimer {len(rows_to_del)} ligne(s)"):
                hist = hist.drop(rows_to_del)
                hist.to_csv('bets_history.csv', index=False)
                st.success("Suppression effectuée.")
                time.sleep(0.5)
                st.rerun()

# --- TAB 4 : MAINTENANCE (Avec Dates) ---
with tab4:
    st.header("État du Système")
    
    col_info, col_action = st.columns(2)
    
    with col_info:
        st.info(f"🧠 **Cerveau (Modèle)** : {get_last_modified_time('nba_predictor.json')}")
        st.info(f"📂 **Données (CSV)** : {get_last_modified_time('nba_games_ready.csv')}")
        st.info(f"📝 **Historique** : {get_last_modified_time('bets_history.csv')}")

    with col_action:
        st.write("Si les dates à gauche semblent vieilles (> 7 jours), lance un entraînement.")
        if st.button("Lancer l'Entraînement Hebdo"):
            status = st.status("Mise à jour de l'intelligence...", expanded=True)
            if run_script_step('train_nba.py', "XGBoost Training", status):
                run_script_step('features_nba.py', "Recalcul Stats", status)
                load_model_resource.clear()
                status.update(label="IA à jour !", state="complete", expanded=False)
                time.sleep(1)
                st.rerun()