import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
import os
import sys
import subprocess
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2
from src import train_nba

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NBA | AGENT PREDiKTOR", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    .match-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #3e3e3e;
        text-align: center;
        margin-bottom: 15px;
    }
    .team-label { color: #aaa; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
    .score-txt { font-size: 1.8em; font-weight: bold; margin: 5px 0; }
    .win-green { color: #4ade80; }
    .win-red { color: #f87171; }
    .section-title { font-size: 1.2em; font-weight: bold; margin-top: 20px; margin-bottom: 10px; color: #fff; border-left: 4px solid #00d4ff; padding-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSIONS ---
if 'schedule_data' not in st.session_state: st.session_state['schedule_data'] = {}
if 'last_update' not in st.session_state: st.session_state['last_update'] = None

DATA_DIR = "data"
MODEL_DIR = "models"
LOGOS_DIR = "assets/logos"
APP_LOGO = "assets/app_logo.png"
HISTORY_FILE = os.path.join(DATA_DIR, "bets_history.csv")
GAMES_FILE = os.path.join(DATA_DIR, "nba_games_ready.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "nba_predictor.json")

# --- 3. FONCTIONS ---

def get_teams_dict():
    nba_teams = teams.get_teams()
    return {t['id']: {'full': t['full_name'], 'code': t['abbreviation'], 'nick': t['nickname']} for t in nba_teams}

TEAMS_DB = get_teams_dict()

@st.cache_resource
def load_resources():
    model = xgb.XGBClassifier()
    df = None
    try:
        if os.path.exists(MODEL_FILE): model.load_model(MODEL_FILE)
        else: model = None
        if os.path.exists(GAMES_FILE):
            df = pd.read_csv(GAMES_FILE)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    except: pass
    return model, df

def show_logo(team_id, width=50):
    path = f"{LOGOS_DIR}/{team_id}.svg"
    if os.path.exists(path): st.image(path, width=width)
    else: st.write("🏀")

def run_script(path, desc, container):
    container.info(f"⚙️ {desc}...", icon="⏳")
    try:
        subprocess.run([sys.executable, path], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        container.error(f"Erreur {desc}")
        st.code(e.stderr)
        return False

def get_clean_name(name_input):
    name_str = str(name_input)
    parts = name_str.split(' ')
    if len(parts) > 0:
        code = parts[0]
        for tid, info in TEAMS_DB.items():
            if info['code'] == code: return info['full']
    return name_str

def get_short_code(name_full):
    for tid, info in TEAMS_DB.items():
        if info['full'] == name_full: return info['code']
    return str(name_full)[:3].upper()

def get_prediction(model, df_history, h_id, a_id):
    if df_history is None or model is None: return None, None
    h_games = df_history[df_history['TEAM_ID'] == h_id].sort_values('GAME_DATE')
    a_games = df_history[df_history['TEAM_ID'] == a_id].sort_values('GAME_DATE')
    if h_games.empty or a_games.empty: return None, None

    lh, la = h_games.iloc[-1], a_games.iloc[-1]
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    rh, ra = (today - lh['GAME_DATE']).days, (today - la['GAME_DATE']).days
    
    row = pd.DataFrame([{
        'EFG_PCT_LAST_5_HOME': lh['EFG_PCT_LAST_5'], 'EFG_PCT_LAST_5_AWAY': la['EFG_PCT_LAST_5'],
        'TOV_PCT_LAST_5_HOME': lh['TOV_PCT_LAST_5'], 'TOV_PCT_LAST_5_AWAY': la['TOV_PCT_LAST_5'],
        'ORB_RAW_LAST_5_HOME': lh['ORB_RAW_LAST_5'], 'ORB_RAW_LAST_5_AWAY': la['ORB_RAW_LAST_5'],
        'DIFF_EFG': lh['EFG_PCT_LAST_5'] - la['EFG_PCT_LAST_5'],
        'DIFF_TOV': lh['TOV_PCT_LAST_5'] - la['TOV_PCT_LAST_5'],
        'DIFF_ORB': lh['ORB_RAW_LAST_5'] - la['ORB_RAW_LAST_5'],
        'DIFF_WIN': lh['WIN_LAST_5'] - la['WIN_LAST_5'],
        'DIFF_REST': min(rh, 7) - min(ra, 7)
    }])
    
    return model.predict_proba(row)[0][1], {'rh': rh, 'ra': ra}

def save_bet_auto(date, h_name, a_name, w_name, conf):
    if not os.path.exists(HISTORY_FILE): 
        with open(HISTORY_FILE, 'w') as f: f.write("Date,Home,Away,Predicted_Winner,Confidence,Type,Result,Real_Winner,User_Prediction,User_Result\n")
    try:
        df = pd.read_csv(HISTORY_FILE)
        # Check doublon strict
        if not df[(df['Date'] == date) & (df['Home'] == h_name) & (df['Away'] == a_name)].empty: return
    except: pass
    # Ajout ligne avec champs vides pour User
    with open(HISTORY_FILE, 'a') as f:
        f.write(f"\n{date},{h_name},{a_name},{w_name},{conf:.1f}%,Auto,,,")

def save_user_vote(date_str, h_name, a_name, user_choice):
    """Sauvegarde le choix de l'utilisateur dans le CSV"""
    if not os.path.exists(HISTORY_FILE): return
    
    try:
        df = pd.read_csv(HISTORY_FILE)
        if 'User_Prediction' not in df.columns: df['User_Prediction'] = None
        
        # Trouver la ligne
        mask = (df['Date'] == date_str) & (df['Home'] == h_name) & (df['Away'] == a_name)
        
        if df[mask].empty:
            st.error("Erreur: Match introuvable.")
            return

        idx = df[mask].index[0]
        df.at[idx, 'User_Prediction'] = user_choice
        df.to_csv(HISTORY_FILE, index=False)
        st.toast(f"Pari enregistré : {user_choice} !", icon="🗳️")
        
    except Exception as e:
        st.error(f"Erreur sauvegarde : {e}")

def get_last_mod(filepath):
    if os.path.exists(filepath):
        return datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%d/%m %H:%M')
    return "N/A"

# FONCTION CLÉ : Scan des jours
def scan_schedule(days_to_check=7):
    found_days = {}
    check_date = datetime.now()
    count_found = 0
    
    for _ in range(days_to_check):
        str_date = check_date.strftime('%Y-%m-%d')
        day_games_list = []
        
        # A. API
        try:
            board = scoreboardv2.ScoreboardV2(game_date=str_date)
            raw = board.game_header.get_data_frame()
            clean = raw.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
            if not clean.empty: day_games_list.append(clean)
        except: pass
        
        # B. HISTORIQUE (Pour les manuels)
        if os.path.exists(HISTORY_FILE):
            try:
                hist = pd.read_csv(HISTORY_FILE)
                manual_today = hist[(hist['Date'] == str_date)]
                if not manual_today.empty: day_games_list.append(manual_today)
            except: pass
        
        if day_games_list:
            found_days[str_date] = day_games_list
            count_found += 1
        
        if count_found >= 2: break
        check_date += timedelta(days=1)
        
    return found_days

# --- INIT ---
model, df_stats = load_resources()
id_to_name = {k: v['full'] for k, v in TEAMS_DB.items()}

# --- HEADER ---
c_head1, c_head2 = st.columns([1, 6])
with c_head1:
    if os.path.exists(APP_LOGO): st.image(APP_LOGO, width=100)
    else: st.title("🏀")
with c_head2:
    st.markdown("## NBA | AGENT PREDiKTOR")
    st.caption("Intelligence Artificielle d'aide à la décision")

# --- NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["📅 MATCHS", "📈 STATS", "🛡️ ADMIN"])

# ==============================================================================
# TAB 1 : MATCHS (AUTO SCAN)
# ==============================================================================
with tab1:
    
    # 1. AUTO SCAN AU DEMARRAGE (Si vide)
    if not st.session_state['schedule_data']:
        with st.spinner("Recherche des matchs à venir..."):
            st.session_state['schedule_data'] = scan_schedule()
            st.session_state['last_update'] = datetime.now().strftime('%H:%M')

    # INFO BARRE
    if st.session_state['last_update']:
        st.caption(f"Dernier scan : {st.session_state['last_update']}")

    # 3. AFFICHAGE DES MATCHS
    schedule = st.session_state.get('schedule_data', {})
    
    if schedule:
        # On charge l'historique UNE FOIS pour la cohérence
        hist_df = pd.DataFrame()
        if os.path.exists(HISTORY_FILE):
            hist_df = pd.read_csv(HISTORY_FILE)

        for date_key, dfs_list in schedule.items():
            is_today = date_key == datetime.now().strftime('%Y-%m-%d')
            # FORMAT DATE : DD.MM.YYYY
            try:
                date_obj = datetime.strptime(date_key, '%Y-%m-%d')
                date_fmt = date_obj.strftime('%d.%m.%Y')
            except: date_fmt = date_key
            
            titre = f"Affiches de ce soir ({date_fmt})" if is_today else f"Affiches du {date_fmt}"
            
            st.markdown(f"<div class='section-title'>{titre}</div>", unsafe_allow_html=True)
            
            cols = st.columns(2)
            card_count = 0
            seen_matches = [] 

            for df_source in dfs_list:
                for index, row in df_source.iterrows():
                    
                    h_id, a_id = 0, 0
                    h_name, a_name = "", ""
                    prob, det = None, None
                    source_type = "API"
                    
                    # 1. RECUPERATION DONNEES
                    if 'HOME_TEAM_ID' in row: # API
                        h_id, a_id = row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']
                        if h_id in TEAMS_DB: h_name = TEAMS_DB[h_id]['full']
                        if a_id in TEAMS_DB: a_name = TEAMS_DB[a_id]['full']
                    elif 'Home' in row: # MANUEL
                        source_type = "History"
                        h_name, a_name = row['Home'], row['Away']
                        h_id = next((k for k, v in TEAMS_DB.items() if v['full'] == get_clean_name(h_name)), 0)
                        a_id = next((k for k, v in TEAMS_DB.items() if v['full'] == get_clean_name(a_name)), 0)

                    match_id = f"{h_name} vs {a_name}"
                    if match_id in seen_matches: continue
                    seen_matches.append(match_id)

                    # 2. LOGIQUE INTELLIGENTE (IA + USER)
                    existing_bet = pd.DataFrame()
                    user_bet_val = None
                    
                    if not hist_df.empty:
                        existing_bet = hist_df[
                            (hist_df['Date'] == date_key) & 
                            (hist_df['Home'] == h_name) & 
                            (hist_df['Away'] == a_name)
                        ]
                    
                    # A. Récupération Prono IA existant
                    if not existing_bet.empty:
                        saved_row = existing_bet.iloc[0]
                        winner = saved_row['Predicted_Winner']
                        conf_str = str(saved_row['Confidence']).replace('%', '')
                        # Récup choix User s'il existe
                        if 'User_Prediction' in saved_row and pd.notna(saved_row['User_Prediction']):
                            user_bet_val = saved_row['User_Prediction']
                            
                        try:
                            conf_val = float(conf_str)/100
                            is_h_win = (winner == h_name)
                            prob = conf_val if is_h_win else (1-conf_val)
                        except: prob = None
                    
                    # B. Si pas de prono IA, on le calcule et on l'enregistre
                    else:
                        if h_id != 0:
                            prob, det = get_prediction(model, df_stats, h_id, a_id)
                            if prob:
                                w = h_name if prob > 0.5 else a_name
                                c = prob*100 if prob > 0.5 else (1-prob)*100
                                save_bet_auto(date_key, h_name, a_name, w, c)
                                st.rerun()

                    # 3. RENDU VISUEL (NOUVEAU DESIGN)
                    if prob is not None and h_id != 0:
                        with cols[card_count % 2]:
                            with st.container(border=True):
                                # -- CALCULS --
                                is_h_win = prob > 0.5
                                ia_choice_name = h_name if is_h_win else a_name
                                ia_conf = prob*100 if is_h_win else (1-prob)*100
                                ia_short = TEAMS_DB.get(h_id if is_h_win else a_id, {}).get('code', ia_choice_name[:3].upper())
                                
                                # -- LAYOUT --
                                c_h, c_vs, c_a = st.columns([2, 1, 2])
                                
                                # Equipe Domicile
                                with c_h:
                                    show_logo(h_id, width=55)
                                    st.caption(TEAMS_DB.get(h_id, {}).get('nick', h_name))
                                
                                # Centre (Info IA)
                                with c_vs:
                                    st.markdown(f"<div style='text-align:center; font-size:0.8em; color:#888;'>IA</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:1.1em; color:#00d4ff;'>{ia_short}</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='text-align:center; font-size:0.8em;'>{ia_conf:.0f}%</div>", unsafe_allow_html=True)

                                # Equipe Extérieur
                                with c_a:
                                    show_logo(a_id, width=55)
                                    st.caption(TEAMS_DB.get(a_id, {}).get('nick', a_name))

                                st.divider()
                                
                                # -- BOUTONS UTILISATEUR --
                                st.caption("Votre vote :")
                                b_col1, b_col2 = st.columns(2)
                                
                                # Bouton Home
                                type_h = "primary" if user_bet_val == h_name else "secondary"
                                if b_col1.button(f"{TEAMS_DB.get(h_id, {}).get('code', 'Home')}", key=f"btn_h_{match_id}_{date_key}", type=type_h, use_container_width=True):
                                    save_user_vote(date_key, h_name, a_name, h_name)
                                    st.rerun()
                                
                                # Bouton Away
                                type_a = "primary" if user_bet_val == a_name else "secondary"
                                if b_col2.button(f"{TEAMS_DB.get(a_id, {}).get('code', 'Away')}", key=f"btn_a_{match_id}_{date_key}", type=type_a, use_container_width=True):
                                    save_user_vote(date_key, h_name, a_name, a_name)
                                    st.rerun()

                        card_count += 1

    elif st.session_state['schedule_data'] == {}:
        st.warning("Aucun match trouvé pour les 7 prochains jours.")

    st.write("")
    st.write("")

    # 4. BILAN FLASH
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        finished = hist[hist['Result'].isin(['GAGNE', 'PERDU'])].copy()
        if not finished.empty:
            st.markdown("<div class='section-title'>Derniers Résultats</div>", unsafe_allow_html=True)
            dates = sorted(finished['Date'].unique(), reverse=True)[:2]
            for d in dates:
                day_rows = finished[finished['Date'] == d]
                wins = len(day_rows[day_rows['Result'] == 'GAGNE'])
                
                try: d_fmt = datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')
                except: d_fmt = d
                
                with st.expander(f"📅 {d_fmt} ({wins}/{len(day_rows)})", expanded=True):
                    for _, r in day_rows.iterrows():
                        icon = "✅" if r['Result'] == "GAGNE" else "❌"
                        h_s = get_short_code(get_clean_name(r['Home']))
                        a_s = get_short_code(get_clean_name(r['Away']))
                        p_s = get_short_code(get_clean_name(r['Predicted_Winner']))
                        st.markdown(f"**{icon} {h_s} vs {a_s}** -> {p_s} ({r['Confidence']})")

# ==============================================================================
# TAB 2 : STATS
# ==============================================================================
with tab2:
    st.markdown("<div class='section-title'>Historique Complet</div>", unsafe_allow_html=True)
    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        if 'Real_Winner' not in df_hist.columns: df_hist['Real_Winner'] = "En attente..."
        df_hist = df_hist.fillna("")
        
        # FIX DATE FORMAT
        df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
        
        df_hist['Home_Clean'] = df_hist['Home'].apply(get_clean_name)
        df_hist['Away_Clean'] = df_hist['Away'].apply(get_clean_name)
        df_hist['Prono_Short'] = df_hist['Predicted_Winner'].apply(lambda x: get_short_code(get_clean_name(x)))
        df_hist['Winner_Short'] = df_hist['Real_Winner'].apply(lambda x: get_short_code(get_clean_name(x)) if x not in ["En attente...", "None", ""] else "...")
        
        # --- GESTION USER ---
        if 'User_Prediction' not in df_hist.columns: df_hist['User_Prediction'] = ""
        df_hist['User_Short'] = df_hist['User_Prediction'].apply(lambda x: get_short_code(get_clean_name(x)) if pd.notna(x) and x != "" else "-")

        display_df = df_hist[['Date', 'Home_Clean', 'Away_Clean', 'Prono_Short', 'User_Short', 'Winner_Short', 'Result', 'Confidence', 'Type']].copy()
        display_df.columns = ['Date', 'Home', 'Away', 'IA', 'Moi', 'Vainqueur', 'Result', 'Conf', 'Type']
        
        display_df = display_df.sort_index(ascending=False)
        display_df.insert(len(display_df.columns), "Del", False)
        
        edited = st.data_editor(
            display_df,
            column_config={
                "Del": st.column_config.CheckboxColumn("🗑️", width="small"),
                "Date": st.column_config.DateColumn("Date", format="DD.MM.YYYY"),
                "IA": st.column_config.TextColumn("IA", help="Choix IA"),
                "Moi": st.column_config.TextColumn("Moi", help="Votre Choix"),
                "Result": st.column_config.TextColumn("Res"),
                "Conf": st.column_config.TextColumn("Conf"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.write("")
        c1, c2 = st.columns(2)
        if c1.button("Supprimer la sélection"):
            to_del_idx = edited[edited.Del == True].index
            if not to_del_idx.empty:
                hist_new = df_hist.drop(to_del_idx)
                cols_save = ['Date', 'Home', 'Away', 'Predicted_Winner', 'Confidence', 'Type', 'Result', 'Real_Winner', 'User_Prediction', 'User_Result']
                # On remet la date en string YYYY-MM-DD
                hist_new['Date'] = hist_new['Date'].dt.strftime('%Y-%m-%d')
                hist_new = hist_new.reindex(columns=cols_save) # Garder ordre colonnes
                hist_new.to_csv(HISTORY_FILE, index=False)
                st.success("Nettoyé !"); time.sleep(0.5); st.rerun()
        if c2.button("Supprimer tous les doublons"):
            df_hist.drop_duplicates(subset=['Date', 'Home', 'Away'], keep='last').to_csv(HISTORY_FILE, index=False)
            st.toast("Doublons nettoyés")
            time.sleep(0.5); st.rerun()

# ==============================================================================
# TAB 3 : ADMIN
# ==============================================================================
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🛠️ Maintenance")
        st.info(f"Cerveau : {get_last_mod(MODEL_FILE)}")
        st.info(f"Données : {get_last_mod(GAMES_FILE)}")
        
        if st.button("Forcer Mise à jour Scores", type="primary"):
            with st.status("Travail en cours...") as s:
                run_script('src/data_nba.py', "Data", s)
                run_script('src/features_nba.py', "Stats", s)
                run_script('src/verify_bets.py', "Verif", s)
                st.session_state['schedule_data'] = {} 
                load_resources.clear()
                s.update(label="Terminé", state="complete")
                st.rerun()
                
        if st.button("Mise à jour Hebdo (Lundi)", use_container_width=True):
            with st.status("Entraînement...") as s:
                succ, msg, acc = train_nba.train_model()
                if succ:
                    run_script('src/features_nba.py', "Stats", s)
                    load_resources.clear()
                    s.update(label=f"Succès ({acc:.1%})", state="complete")
                else: s.error(msg)

    with c2:
        st.subheader("🔮 Ajout Manuel")
        team_names = [f"{v['code']} - {v['full']}" for k,v in TEAMS_DB.items()]
        hm = st.selectbox("Domicile", team_names, index=None)
        aw = st.selectbox("Extérieur", team_names, index=None)
        dt = st.date_input("Date du match", value=datetime.now())
        if hm and aw:
            if st.button("Analyser & Ajouter"):
                h_code = hm.split(' - ')[0]
                a_code = aw.split(' - ')[0]
                h_id = next(k for k,v in TEAMS_DB.items() if v['code'] == h_code)
                a_id = next(k for k,v in TEAMS_DB.items() if v['code'] == a_code)
                prob, _ = get_prediction(model, df_stats, h_id, a_id)
                if prob:
                    win_name = TEAMS_DB[h_id]['full'] if prob > 0.5 else TEAMS_DB[a_id]['full']
                    conf = prob*100 if prob > 0.5 else (1-prob)*100
                    st.success(f"Vainqueur : {win_name} ({conf:.1f}%)")
                    save_bet_auto(dt.strftime('%Y-%m-%d'), TEAMS_DB[h_id]['full'], TEAMS_DB[a_id]['full'], win_name, conf)
                    st.session_state['schedule_data'] = {} # Force refresh
                    st.rerun()