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
st.set_page_config(page_title="NBA Manager v3.2", page_icon="🏀", layout="wide")

# --- MEMOIRE DE SESSION ---
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
    # Format : "OKC - Thunder"
    return {f"{t['abbreviation']} - {t['nickname']}": t['id'] for t in nba_teams}

def get_last_modified_time(filename):
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
    
    try:
        current_df = pd.read_csv('bets_history.csv')
        today = datetime.now().strftime('%Y-%m-%d')
        # On vérifie les doublons sur la date et les équipes
        exists = not current_df[(current_df['Date'] == today) & (current_df['Home'] == home) & (current_df['Away'] == away)].empty
        if exists: return 
    except:
        pass

    with open('bets_history.csv', 'a') as f:
        date = datetime.now().strftime('%Y-%m-%d')
        f.write(f"\n{date},{home},{away},{winner},{conf:.1f}%,{type_bet},")

# --- INTERFACE ---

st.title("🏀 NBA Manager v3.2")

model = load_model_resource()
df = load_data_resource()
teams_dict = get_team_list()

# --- COSMÉTIQUE 1 : NOMS COMPLETS ---
# On remplace le tiret par un espace pour avoir "OKC Thunder"
id_to_name = {v: k.replace(' - ', ' ') for k, v in teams_dict.items()}

tab1, tab2, tab3, tab4 = st.tabs(["🌞 Matchs du Jour", "🔮 Manuel", "📊 Bilan", "⚙️ Maintenance"])

# --- TAB 1 : AUTO PREDICT ---
with tab1:
    st.header("Routine Matinale")
    
    if st.session_state['last_run_date']:
        st.caption(f"Dernière analyse : {st.session_state['last_run_date']}")

    if st.button("🚀 LANCER LA JOURNEE", type="primary"):
        status_box = st.status("Traitement en cours...", expanded=True)
        
        if run_script_step('data_nba.py', "Mise a jour Donnees", status_box):
            if run_script_step('features_nba.py', "Calcul Stats", status_box):
                run_script_step('verify_bets.py', "Verification Paris", status_box)
                
                load_data_resource.clear()
                df = load_data_resource()
                
                status_box.write("🔎 Recherche matchs du soir...")
                try:
                    board = scoreboardv2.ScoreboardV2(game_date=datetime.now().strftime('%Y-%m-%d'))
                    games_raw = board.game_header.get_data_frame()
                    games_clean = games_raw.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
                    
                    st.session_state['games_today'] = games_clean
                    st.session_state['last_run_date'] = datetime.now().strftime('%H:%M:%S')
                    status_box.update(label="Terminé !", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Erreur API : {e}")

    st.divider()

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
                        save_bet(h_name, a_name, win, conf, "Auto")
                st.divider()
        else:
            st.info("Aucun match trouvé.")
    else:
        st.write("En attente de lancement...")

# --- TAB 2 : MANUEL ---
with tab2:
    if model is None or df is None:
        st.warning("Donnees manquantes.")
    else:
        c1, c2 = st.columns(2)
        with c1: h_choice = st.selectbox("Domicile", list(teams_dict.keys()), index=None, placeholder="Choisis l'équipe...")
        with c2: a_choice = st.selectbox("Extérieur", list(teams_dict.keys()), index=None, placeholder="Choisis l'équipe...")
            
        if h_choice and a_choice:
            if st.button("Analyser le Duel"):
                h_id, a_id = teams_dict[h_choice], teams_dict[a_choice]
                # Nom propre pour affichage
                h_nice = h_choice.replace(' - ', ' ')
                a_nice = a_choice.replace(' - ', ' ')
                
                prob, d = get_prediction(model, df, h_id, a_id)
                if prob is not None:
                    if prob > 0.5:
                        win, conf = h_nice, prob*100
                        st.success(f"🏆 {win} ({conf:.1f}%)")
                    else:
                        win, conf = a_nice, (1-prob)*100
                        st.success(f"🏆 {win} ({conf:.1f}%)")
                    save_bet(h_nice, a_nice, win, conf, "Manual")

# --- TAB 3 : BILAN (REFACTORISÉ) ---
with tab3:
    st.header("Historique")
    if os.path.exists('bets_history.csv'):
        # 1. Chargement et Préparation
        hist = pd.read_csv('bets_history.csv')
        hist_sorted = hist.sort_index(ascending=False) # Plus récents en haut
        
        # 2. Ajout colonne de sélection
        # On utilise st.data_editor pour avoir des cases à cocher
        # On crée une copie pour l'édition avec une colonne 'Select' à True/False
        hist_sorted.insert(0, "Select", False)
        
        # --- COSMÉTIQUE 3 : TABLEAU INTERACTIF ---
        edited_df = st.data_editor(
            hist_sorted,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Sél.",
                    help="Coche pour supprimer",
                    default=False,
                )
            },
            disabled=["Date", "Home", "Away", "Predicted_Winner", "Confidence", "Type", "Result"],
            hide_index=True,
        )
        
        # 3. Zone de nettoyage discrète
        st.write("")
        with st.expander("🗑️ Zone de nettoyage (Doublons & Suppression)"):
            c_clean1, c_clean2 = st.columns(2)
            
            with c_clean1:
                if st.button("Supprimer la sélection"):
                    # On récupère les lignes cochées
                    to_delete = edited_df[edited_df.Select == True]
                    if not to_delete.empty:
                        # On supprime du dataframe original en utilisant les index (qui sont cachés mais existent)
                        # Attention: edited_df a les mêmes index que hist_sorted
                        indices_to_drop = to_delete.index
                        hist_final = hist.drop(indices_to_drop)
                        
                        hist_final.to_csv('bets_history.csv', index=False)
                        st.success(f"{len(to_delete)} ligne(s) supprimée(s).")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("Aucune ligne cochée.")

            with c_clean2:
                if st.button("🧹 Nettoyer tous les doublons auto"):
                    before = len(hist)
                    hist = hist.drop_duplicates(subset=['Date', 'Home', 'Away'], keep='last')
                    hist.to_csv('bets_history.csv', index=False)
                    after = len(hist)
                    st.toast(f"{before - after} doublons supprimés !", icon="✨")
                    time.sleep(1)
                    st.rerun()

        # KPI
        st.divider()
        vals = hist[hist['Result'].isin(['GAGNE', 'PERDU'])]
        if not vals.empty:
            wins = len(vals[vals['Result']=='GAGNE'])
            acc = (wins/len(vals))*100
            st.metric("Précision Globale", f"{acc:.1f}%", f"{len(vals)} paris terminés")

# --- TAB 4 : MAINTENANCE ---
with tab4:
    st.header("État du Système")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🧠 Modèle : {get_last_modified_time('nba_predictor.json')}")
        st.info(f"📝 Historique : {get_last_modified_time('bets_history.csv')}")
    with col2:
        st.write("Mise à jour hebdo recommandée le Lundi.")
        if st.button("Lancer l'Entraînement Hebdo"):
            status = st.status("Mise à jour...", expanded=True)
            if run_script_step('train_nba.py', "XGBoost Training", status):
                run_script_step('features_nba.py', "Recalcul Stats", status)
                load_model_resource.clear()
                status.update(label="IA à jour !", state="complete", expanded=False)
                time.sleep(1)
                st.rerun()