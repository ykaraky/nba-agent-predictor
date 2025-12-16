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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NBA Manager v4.2", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    .match-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MEMOIRE DE SESSION ---
if 'games_today' not in st.session_state: st.session_state['games_today'] = None
if 'last_run_date' not in st.session_state: st.session_state['last_run_date'] = None

# --- 3. FONCTIONS ---

@st.cache_resource
def load_model_resource():
    model = xgb.XGBClassifier()
    try:
        if os.path.exists("nba_predictor.json"):
            model.load_model("nba_predictor.json")
            return model
        return None
    except: return None

@st.cache_data
def load_data_resource():
    try:
        if os.path.exists('nba_games_ready.csv'):
            df = pd.read_csv('nba_games_ready.csv')
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            return df
        return None
    except: return None

def get_team_list():
    nba_teams = teams.get_teams()
    return {f"{t['abbreviation']} - {t['nickname']}": t['id'] for t in nba_teams}

def show_team_logo(team_id, width=60):
    logo_path = f"logos/{team_id}.svg"
    if os.path.exists(logo_path):
        st.image(logo_path, width=width)
    else:
        st.write(f"ID: {team_id}")

def get_last_modified_time(filename):
    if os.path.exists(filename):
        timestamp = os.path.getmtime(filename)
        return datetime.fromtimestamp(timestamp).strftime('%d/%m à %H:%M')
    return "Jamais"

def run_script_step(script_name, step_name, status_container):
    try:
        status_container.write(f"⏳ {step_name}...")
        subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        status_container.error(f"Erreur : {step_name}")
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
        exists = not current_df[(current_df['Date'] == today) & (current_df['Home'] == home) & (current_df['Away'] == away)].empty
        if exists: return 
    except: pass

    with open('bets_history.csv', 'a') as f:
        date = datetime.now().strftime('%Y-%m-%d')
        f.write(f"\n{date},{home},{away},{winner},{conf:.1f}%,{type_bet},")

# --- INIT ---
model = load_model_resource()
df = load_data_resource()
teams_dict = get_team_list()
id_to_name = {v: k.replace(' - ', ' ') for k, v in teams_dict.items()}

# --- INTERFACE ---

st.title("🏀 NBA Manager v4.2")

tab1, tab2, tab3, tab4 = st.tabs(["🌞 Matchs", "📊 Bilan", "🔮 Manuel", "⚙️ Admin"])

# --- TAB 1 : DASHBOARD ---
with tab1:
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # 1. Chargement Intelligent (Priorité à l'historique existant)
    if os.path.exists('bets_history.csv'):
        try:
            hist_check = pd.read_csv('bets_history.csv')
            todays_bets = hist_check[hist_check['Date'] == today_str]
            
            # Si on a des paris pour aujourd'hui (Auto ou Manuel), on les prend !
            if not todays_bets.empty:
                st.session_state['games_today'] = todays_bets
                if st.session_state['last_run_date'] is None:
                    st.session_state['last_run_date'] = "Données Historique"
        except: pass

    col_btn, col_txt = st.columns([1, 4])
    with col_btn:
        label_btn = "Forcer mise à jour" if st.session_state['last_run_date'] else "Lancer la routine"
        type_btn = "secondary" if st.session_state['last_run_date'] else "primary"
        
        if st.button(label_btn, type=type_btn):
            status = st.status("Mise à jour...", expanded=True)
            if run_script_step('data_nba.py', "Data", status):
                if run_script_step('features_nba.py', "Stats", status):
                    run_script_step('verify_bets.py', "Verif", status)
                    load_data_resource.clear()
                    load_model_resource.clear() # On vide bien les caches
                    df = load_data_resource()
                    model = load_model_resource()
                    try:
                        status.write("API NBA...")
                        board = scoreboardv2.ScoreboardV2(game_date=datetime.now().strftime('%Y-%m-%d'))
                        games_raw = board.game_header.get_data_frame()
                        games_clean = games_raw.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
                        st.session_state['games_today'] = games_clean
                        st.session_state['last_run_date'] = datetime.now().strftime('%H:%M')
                        status.update(label="Prêt !", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:
                        status.update(label="Erreur API", state="error")
                        st.error(f"Détail : {e}")
    with col_txt:
        if st.session_state['last_run_date']:
            st.success(f"✅ Données chargées ({st.session_state['last_run_date']})")

    st.divider()

    if st.session_state['games_today'] is not None and not st.session_state['games_today'].empty:
        st.subheader("🏀 Affiches de la nuit")
        games_df = st.session_state['games_today']
        cols = st.columns(2)
        
        for index, row in games_df.iterrows():
            col_idx = index % 2
            with cols[col_idx]:
                with st.container(border=True):
                    
                    # CAS A : C'est un match API Live (pas encore parié)
                    if 'HOME_TEAM_ID' in row:
                        h_id, a_id = row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']
                        h_name = id_to_name.get(h_id, str(h_id))
                        a_name = id_to_name.get(a_id, str(a_id))
                        prob, det = get_prediction(model, df, h_id, a_id)
                        
                        if prob is not None:
                            # Calcul et Sauvegarde immédiate
                            w = h_name if prob > 0.5 else a_name
                            c = prob*100 if prob > 0.5 else (1-prob)*100
                            save_bet(h_name, a_name, w, c, "Auto")
                            
                            # Affichage Live
                            c_home, c_mid, c_away = st.columns([1, 2, 1])
                            with c_home: show_team_logo(h_id); st.caption(h_name.split(' ')[-1])
                            with c_away: show_team_logo(a_id); st.caption(a_name.split(' ')[-1])
                            with c_mid:
                                col = "green" if prob > 0.5 else "#d9534f"
                                txt = f"👈 {c:.0f}%" if prob > 0.5 else f"{c:.0f}% 👉"
                                st.markdown(f"<h2 style='text-align: center; color: {col}; margin:0;'>{txt}</h2>", unsafe_allow_html=True)

                    # CAS B : C'est un match HISTORIQUE (Déjà parié ou Manuel)
                    else:
                        # On récupère les valeurs DÉJÀ ENREGISTRÉES (Pas de recalcul !)
                        h_name = row['Home']
                        a_name = row['Away']
                        winner = row['Predicted_Winner']
                        conf_str = str(row['Confidence']).replace('%', '')
                        
                        # Retrouver les IDs pour les logos
                        name_to_id = {v: k for k, v in id_to_name.items()}
                        h_id = name_to_id.get(h_name, 0)
                        a_id = name_to_id.get(a_name, 0)
                        
                        c_home, c_mid, c_away = st.columns([1, 2, 1])
                        with c_home: show_team_logo(h_id); st.caption(h_name.split(' ')[-1])
                        with c_away: show_team_logo(a_id); st.caption(a_name.split(' ')[-1])
                        with c_mid:
                            # Affichage basé sur le Vainqueur enregistré
                            is_home_winner = (winner == h_name)
                            col = "green" if is_home_winner else "#d9534f"
                            try:
                                val = float(conf_str)
                                txt = f"👈 {val:.0f}%" if is_home_winner else f"{val:.0f}% 👉"
                            except:
                                txt = winner # Fallback si erreur de format
                                
                            st.markdown(f"<h2 style='text-align: center; color: {col}; margin:0;'>{txt}</h2>", unsafe_allow_html=True)
                            st.caption(f"Type: {row['Type']}")

    elif st.session_state['games_today'] is not None:
        st.info("Aucun match ce soir.")
    else:
        st.write("En attente...")

# --- TAB 2 : BILAN (CORRIGÉ) ---
with tab2:
    if os.path.exists('bets_history.csv'):
        hist = pd.read_csv('bets_history.csv')
        
        if 'Real_Winner' not in hist.columns: hist['Real_Winner'] = "En attente..."
        
        hist_sorted = hist.sort_index(ascending=False)
        hist_sorted.insert(len(hist_sorted.columns), "Select", False)
        
        # --- CONFIGURATION STRICTE TEXTE ---
        col_cfg = {
            "Select": st.column_config.CheckboxColumn("🗑️", width="small"),
            "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
            "Predicted_Winner": st.column_config.TextColumn("Prono IA"),
            "Real_Winner": st.column_config.TextColumn("Vainqueur Réel"),
            "Result": st.column_config.TextColumn("Résultat", width="small"),
            # ICI : On force l'affichage en TEXTE simple
            "Confidence": st.column_config.TextColumn("Confiance"), 
        }
        
        cols_order = ["Date", "Home", "Away", "Predicted_Winner", "Real_Winner", "Result", "Confidence", "Type", "Select"]

        edited_df = st.data_editor(
            hist_sorted,
            column_config=col_cfg,
            column_order=cols_order,
            hide_index=True,
            disabled=[c for c in cols_order if c != "Select"],
        )
        
        with st.expander("Outils de nettoyage"):
            c1, c2 = st.columns(2)
            if c1.button("Supprimer la sélection"):
                rows_to_delete = edited_df[edited_df.Select == True]
                if not rows_to_delete.empty:
                    original_csv = pd.read_csv('bets_history.csv')
                    for index, row in rows_to_delete.iterrows():
                        original_csv = original_csv[
                            ~((original_csv['Date'] == row['Date']) & 
                              (original_csv['Home'] == row['Home']) & 
                              (original_csv['Away'] == row['Away']))
                        ]
                    original_csv.to_csv('bets_history.csv', index=False)
                    st.success("Fait."); time.sleep(0.5); st.rerun()
            if c2.button("Supprimer doublons"):
                hist.drop_duplicates(subset=['Date','Home','Away'], keep='last').to_csv('bets_history.csv', index=False)
                st.rerun()
        
        done = hist[hist['Result'].isin(['GAGNE', 'PERDU'])]
        if not done.empty:
            wins = len(done[done['Result']=='GAGNE'])
            st.caption(f"Précision: {wins/len(done):.1%} ({wins}/{len(done)})")

# --- TAB 3 : MANUEL ---
with tab3:
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1: h_c = st.selectbox("Domicile", list(teams_dict.keys()), index=None, key="m_h")
    with c3: a_c = st.selectbox("Extérieur", list(teams_dict.keys()), index=None, key="m_a")
    
    if h_c and a_c:
        with c2:
            st.write("")
            st.write("")
            if st.button("VS", type="primary", use_container_width=True):
                h_id, a_id = teams_dict[h_c], teams_dict[a_c]
                prob, det = get_prediction(model, df, h_id, a_id)
                if prob:
                    st.divider()
                    cc1, cc2, cc3 = st.columns([1,2,1])
                    with cc1: 
                        show_team_logo(h_id)
                        st.caption(h_c.split(' - ')[-1])
                    with cc3: 
                        show_team_logo(a_id)
                        st.caption(a_c.split(' - ')[-1])
                    with cc2: 
                        w_txt = f"{(prob*100):.0f}%" if prob>0.5 else f"{(1-prob)*100:.0f}%"
                        col = "green" if prob>0.5 else "red"
                        arrow = "👈" if prob > 0.5 else "👉"
                        st.markdown(f"<h2 style='text-align:center; color:{col}'>{arrow} {w_txt}</h2>", unsafe_allow_html=True)
                        w_name = h_c.replace(' - ',' ') if prob > 0.5 else a_c.replace(' - ',' ')
                        save_bet(h_c.replace(' - ',' '), a_c.replace(' - ',' '), w_name, prob*100 if prob>0.5 else (1-prob)*100, "Manual")

# --- TAB 4 : ADMIN ---
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Modèle : {get_last_modified_time('nba_predictor.json')}")
        st.info(f"Données : {get_last_modified_time('nba_games_ready.csv')}")
    with col2:
        if st.button("Mise à jour Hebdo (Lundi)"):
            s = st.status("Travail en cours...")
            run_script_step('train_nba.py', "Training", s)
            run_script_step('features_nba.py', "Stats", s)
            load_model_resource.clear()
            s.update(label="Terminé", state="complete")