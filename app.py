import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
import os
import sys
import subprocess
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2, leaguestandingsv3
from src import train_nba

# --- 1. CONFIGURATION & CSS AVANCÉ ---
st.set_page_config(page_title="NBA | AGENT PREDiKTOR", page_icon="🏀", layout="wide")

# CSS HACKS : Layout, Header, Mobile
st.markdown("""
<style>
    /* 1. Header & Nav Fixed */
    header[data-testid="stHeader"] {
        z-index: 1000;
        background-color: #0e1117;
        opacity: 0.98;
    }
    div[data-testid="stTabs"] {
        position: sticky;
        top: 3rem; 
        background-color: #0e1117;
        z-index: 999;
        padding-top: 0px;
        padding-bottom: 0px;
        border-bottom: 1px solid #333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px; /* Moins haut */
        font-size: 1em;
        font-weight: 600;
        padding-top: 0px;
        padding-bottom: 0px;
    }
    
    /* 2. Cards Styles */
    .match-card-container {
        border: 1px solid #2e3035;
        border-radius: 8px;
        padding: 10px;
        background-color: #1c1f26;
    }
    .team-name { font-weight: bold; font-size: 1.0em; margin: 0; }
    .team-sub { font-size: 0.75em; color: #aaa; }
    
    /* IA Prono Centré */
    .ia-box {
        text-align: center;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .ia-label { font-size: 0.7em; color: #666; text-transform: uppercase; letter-spacing: 1px; }
    .ia-val { font-size: 1.2em; font-weight: 900; color: #00d4ff; }
    
    /* Boutons User Sober (Bleu moins agressif géré par Streamlit theme, ici on force le gris si pas actif) */
    
    /* 3. MOBILE OPTIMIZATIONS (@media) */
    @media (max-width: 640px) {
        /* Forcer le centrage partout sur mobile */
        div[data-testid="column"] {
            text-align: center !important;
            align-items: center !important;
        }
        .team-name { font-size: 0.9em; }
        /* Cacher les logos trop gros sur mobile si besoin */
        img { max-width: 40px !important; }
    }
    
    /* Cacher liens ancres */
    .css-15zrgzn {display: none;}
</style>
""", unsafe_allow_html=True)

# --- 2. SESSIONS ---
if 'schedule_data' not in st.session_state: st.session_state['schedule_data'] = {}
if 'last_update' not in st.session_state: st.session_state['last_update'] = None
# Pour gérer le mode "Modifier" des boutons
if 'edit_modes' not in st.session_state: st.session_state['edit_modes'] = {}

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

@st.cache_data(ttl=3600)
def get_standings_db():
    try:
        standings = leaguestandingsv3.LeagueStandingsV3()
        df = standings.standings.get_data_frame()
        res = {}
        for _, row in df.iterrows():
            tid = row['TeamID']
            streak_val = row['CurrentStreak']
            if isinstance(streak_val, int) or (isinstance(streak_val, str) and streak_val.lstrip('-').isdigit()):
                val = int(streak_val)
                streak_short = f"W{abs(val)}" if val > 0 else f"L{abs(val)}"
            else: streak_short = str(streak_val)

            res[tid] = {
                'rec': row['Record'],
                'strk': streak_short,
                'rank': row['PlayoffRank'],
                'conf': row['Conference']
            }
        return res
    except: return {}

STANDINGS_DB = get_standings_db()

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
        if not df[(df['Date'] == date) & (df['Home'] == h_name) & (df['Away'] == a_name)].empty: return
    except: pass
    with open(HISTORY_FILE, 'a') as f:
        f.write(f"\n{date},{h_name},{a_name},{w_name},{conf:.1f}%,Auto,,,")

def save_user_vote(date_str, h_name, a_name, user_choice):
    if not os.path.exists(HISTORY_FILE): return
    try:
        df = pd.read_csv(HISTORY_FILE)
        if 'User_Prediction' not in df.columns: df['User_Prediction'] = None
        mask = (df['Date'] == date_str) & (df['Home'] == h_name) & (df['Away'] == a_name)
        if df[mask].empty: return
        idx = df[mask].index[0]
        df.at[idx, 'User_Prediction'] = user_choice
        df.to_csv(HISTORY_FILE, index=False)
        # On ferme le mode édition après sauvegarde
        st.session_state['edit_modes'][f"{h_name} vs {a_name}"] = False
    except Exception as e: st.error(f"Erreur : {e}")

def get_last_mod(filepath):
    if os.path.exists(filepath):
        return datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%d/%m %H:%M')
    return "N/A"

def scan_schedule(days_to_check=7):
    found_days = {}
    check_date = datetime.now()
    count_found = 0
    for _ in range(days_to_check):
        str_date = check_date.strftime('%Y-%m-%d')
        day_games_list = []
        try:
            board = scoreboardv2.ScoreboardV2(game_date=str_date)
            raw = board.game_header.get_data_frame()
            clean = raw.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
            if not clean.empty: day_games_list.append(clean)
        except: pass
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

# --- HEADER (Contenu) ---
# Astuce: On utilise un container vide pour le spacing si besoin, mais le CSS gère le fixed
c_head1, c_head2 = st.columns([1, 8])
with c_head1:
    if os.path.exists(APP_LOGO): st.image(APP_LOGO, width=70)
    else: st.title("🏀")
with c_head2:
    st.markdown("<h3 style='margin:0; padding-top:10px;'>NBA AGENT PREDIKTOR</h3>", unsafe_allow_html=True)

# --- NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["MATCHS", "STATS", "ADMIN"])

# ==============================================================================
# TAB 1 : MATCHS
# ==============================================================================
with tab1:
    if not st.session_state['schedule_data']:
        with st.spinner("Chargement..."):
            st.session_state['schedule_data'] = scan_schedule()

    schedule = st.session_state.get('schedule_data', {})
    
    if schedule:
        hist_df = pd.DataFrame()
        if os.path.exists(HISTORY_FILE): hist_df = pd.read_csv(HISTORY_FILE)

        for date_key, dfs_list in schedule.items():
            is_today = date_key == datetime.now().strftime('%Y-%m-%d')
            try: d_fmt = datetime.strptime(date_key, '%Y-%m-%d').strftime('%d.%m.%Y')
            except: d_fmt = date_key
            
            st.markdown(f"#### {'🔥 Ce Soir' if is_today else '📅 ' + d_fmt}")
            
            # --- PREPARATION DES MATCHS DU JOUR ---
            matches_to_display = []
            seen_matches = []
            
            for df_source in dfs_list:
                for index, row in df_source.iterrows():
                    h_id, a_id = 0, 0
                    h_name, a_name = "", ""
                    prob = None
                    
                    if 'HOME_TEAM_ID' in row: 
                        h_id, a_id = row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']
                        if h_id in TEAMS_DB: h_name = TEAMS_DB[h_id]['full']
                        if a_id in TEAMS_DB: a_name = TEAMS_DB[a_id]['full']
                    elif 'Home' in row:
                        h_name, a_name = row['Home'], row['Away']
                        h_id = next((k for k, v in TEAMS_DB.items() if v['full'] == get_clean_name(h_name)), 0)
                        a_id = next((k for k, v in TEAMS_DB.items() if v['full'] == get_clean_name(a_name)), 0)

                    match_id = f"{h_name} vs {a_name}"
                    if match_id in seen_matches: continue
                    seen_matches.append(match_id)

                    # LOGIQUE
                    existing_bet = pd.DataFrame()
                    user_bet_val = None
                    if not hist_df.empty:
                        existing_bet = hist_df[(hist_df['Date'] == date_key) & (hist_df['Home'] == h_name) & (hist_df['Away'] == a_name)]
                    
                    if not existing_bet.empty:
                        saved_row = existing_bet.iloc[0]
                        winner = saved_row['Predicted_Winner']
                        conf_str = str(saved_row['Confidence']).replace('%', '')
                        if 'User_Prediction' in saved_row and pd.notna(saved_row['User_Prediction']):
                            user_bet_val = saved_row['User_Prediction']
                        try:
                            conf_val = float(conf_str)/100
                            is_h_win = (winner == h_name)
                            prob = conf_val if is_h_win else (1-conf_val)
                        except: prob = None
                    else:
                        if h_id != 0:
                            prob, det = get_prediction(model, df_stats, h_id, a_id)
                            if prob:
                                w = h_name if prob > 0.5 else a_name
                                c = prob*100 if prob > 0.5 else (1-prob)*100
                                save_bet_auto(date_key, h_name, a_name, w, c)
                                st.rerun()
                    
                    if prob is not None and h_id != 0:
                        matches_to_display.append({
                            'h_id': h_id, 'a_id': a_id, 'h_name': h_name, 'a_name': a_name,
                            'prob': prob, 'user_bet': user_bet_val, 'mid': match_id, 'date': date_key
                        })

            # --- AFFICHAGE GRID (2 par ligne sur Desktop) ---
            # On utilise une boucle avec index pour gérer les colonnes
            if matches_to_display:
                cols = st.columns(2) # Grid de 2 colonnes
                for i, m in enumerate(matches_to_display):
                    with cols[i % 2]: # Alterne gauche/droite
                        with st.container(border=True): # Card Container
                            
                            # Datas
                            is_h_win = m['prob'] > 0.5
                            ia_conf = m['prob']*100 if is_h_win else (1-m['prob'])*100
                            ia_short = TEAMS_DB.get(m['h_id'] if is_h_win else m['a_id'], {}).get('code', 'IA')
                            
                            inf_h = STANDINGS_DB.get(m['h_id'], {'rec': '', 'strk': '', 'rank': ''})
                            inf_a = STANDINGS_DB.get(m['a_id'], {'rec': '', 'strk': '', 'rank': ''})
                            c_strk_h = "#4ade80" if 'W' in inf_h['strk'] else "#f87171"
                            c_strk_a = "#4ade80" if 'W' in inf_a['strk'] else "#f87171"

                            # LAYOUT INTERNE CARTE
                            c1, c2, c3 = st.columns([3, 2, 3])
                            
                            # TEAM A (Gauche)
                            with c1:
                                st.markdown(f"<div style='text-align:center;'>", unsafe_allow_html=True)
                                show_logo(m['h_id'], width=45)
                                st.markdown(f"<div class='team-name'>{TEAMS_DB.get(m['h_id'],{}).get('code', m['h_name'])}</div>", unsafe_allow_html=True)
                                if inf_h['rec']:
                                    st.markdown(f"<div class='team-sub'>#{inf_h['rank']} ({inf_h['rec']})</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='color:{c_strk_h}; font-size:0.7em; font-weight:bold;'>{inf_h['strk']}</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                            # CENTER (IA + User)
                            with c2:
                                # IA
                                st.markdown(f"<div class='ia-box'><div class='ia-label'>PRONO IA</div><div class='ia-val'>{ia_short}</div><div style='font-size:0.7em; color:#888;'>{ia_conf:.0f}%</div></div>", unsafe_allow_html=True)
                                
                                st.divider()
                                
                                # USER INTERACTION (Safe Mode)
                                has_voted = (m['user_bet'] is not None and m['user_bet'] != "")
                                is_editing = st.session_state['edit_modes'].get(m['mid'], False)
                                
                                if has_voted and not is_editing:
                                    # Affichage vote pris
                                    vote_short = TEAMS_DB.get(next((k for k,v in TEAMS_DB.items() if v['full'] == m['user_bet']),0), {}).get('code', m['user_bet'])
                                    st.markdown(f"<div style='text-align:center; font-size:0.8em;'>Votre choix:<br><b>{vote_short}</b></div>", unsafe_allow_html=True)
                                    if st.button("Modifier", key=f"mod_{m['mid']}"):
                                        st.session_state['edit_modes'][m['mid']] = True
                                        st.rerun()
                                else:
                                    # Mode Boutons
                                    # Style sobre : secondary par defaut.
                                    code_h = TEAMS_DB.get(m['h_id'], {}).get('code', 'H')
                                    code_a = TEAMS_DB.get(m['a_id'], {}).get('code', 'A')
                                    
                                    if st.button(code_h, key=f"bh_{m['mid']}", use_container_width=True):
                                        save_user_vote(m['date'], m['h_name'], m['a_name'], m['h_name'])
                                        st.rerun()
                                    if st.button(code_a, key=f"ba_{m['mid']}", use_container_width=True):
                                        save_user_vote(m['date'], m['h_name'], m['a_name'], m['a_name'])
                                        st.rerun()

                            # TEAM B (Droite)
                            with c3:
                                st.markdown(f"<div style='text-align:center;'>", unsafe_allow_html=True)
                                show_logo(m['a_id'], width=45)
                                st.markdown(f"<div class='team-name'>{TEAMS_DB.get(m['a_id'],{}).get('code', m['a_name'])}</div>", unsafe_allow_html=True)
                                if inf_a['rec']:
                                    st.markdown(f"<div class='team-sub'>#{inf_a['rank']} ({inf_a['rec']})</div>", unsafe_allow_html=True)
                                    st.markdown(f"<div style='color:{c_strk_a}; font-size:0.7em; font-weight:bold;'>{inf_a['strk']}</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state['schedule_data'] == {}:
        st.info("Aucun match à venir détecté.")

    # 4. DERNIERS RESULTATS (Compact & Clean)
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        finished = hist[hist['Result'].isin(['GAGNE', 'PERDU'])].copy()
        
        if not finished.empty:
            st.markdown("---")
            st.caption("🏁 RÉSULTATS RÉCENTS")
            dates = sorted(finished['Date'].unique(), reverse=True)[:2]
            
            for d in dates:
                day_rows = finished[finished['Date'] == d]
                ia_wins = len(day_rows[day_rows['Result'] == 'GAGNE'])
                user_wins = 0
                if 'User_Result' in day_rows.columns:
                    user_wins = len(day_rows[day_rows['User_Result'] == 'GAGNE'])
                total = len(day_rows)
                
                try: d_fmt = datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m')
                except: d_fmt = d
                
                # Header Accordeon compact
                with st.expander(f"📅 {d_fmt} | 🤖 {ia_wins}/{total} | 👤 {user_wins}/{total}", expanded=False):
                    for _, r in day_rows.iterrows():
                        # Ligne ultra-compacte pour Mobile: [MATCH] [WINNER] [IA] [IK]
                        match_lbl = f"{get_short_code(r['Home'])}-{get_short_code(r['Away'])}"
                        win_lbl = get_short_code(r['Real_Winner']) if pd.notna(r['Real_Winner']) else "?"
                        
                        # IA Badge
                        ia_res = "✅" if r['Result'] == "GAGNE" else "❌"
                        # IK Badge
                        ik_res = "➖"
                        if 'User_Result' in r and pd.notna(r['User_Result']):
                            ik_res = "✅" if r['User_Result'] == "GAGNE" else "❌"
                            
                        # Layout Colonnes
                        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
                        c1.caption(match_lbl)
                        c2.caption(f"Win: {win_lbl}")
                        c3.write(f"🤖{ia_res}")
                        c4.write(f"👤{ik_res}")

# ==============================================================================
# TAB 2 : STATS
# ==============================================================================
with tab2:
    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        df_hist = df_hist.fillna("")
        df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
        
        # 1. Clean Data
        df_hist['H'] = df_hist['Home'].apply(lambda x: get_short_code(get_clean_name(x)))
        df_hist['A'] = df_hist['Away'].apply(lambda x: get_short_code(get_clean_name(x)))
        df_hist['Win'] = df_hist['Real_Winner'].apply(lambda x: get_short_code(get_clean_name(x)) if x not in ["En attente...", ""] else "...")
        
        df_hist['IA_P'] = df_hist['Predicted_Winner'].apply(lambda x: get_short_code(get_clean_name(x)))
        # Resultat TEXTE (plus de checkbox)
        df_hist['IA_R'] = df_hist['Result'].apply(lambda x: "WIN" if x == 'GAGNE' else "LOSS" if x == 'PERDU' else "-")
        
        if 'User_Prediction' not in df_hist.columns: df_hist['User_Prediction'] = ""
        df_hist['IK_P'] = df_hist['User_Prediction'].apply(lambda x: get_short_code(get_clean_name(x)) if x != "" else "-")
        
        if 'User_Result' not in df_hist.columns: df_hist['User_Result'] = ""
        df_hist['IK_R'] = df_hist['User_Result'].apply(lambda x: "WIN" if x == 'GAGNE' else "LOSS" if x == 'PERDU' else "-")

        display_df = df_hist[['Date', 'H', 'A', 'Win', 'IK_P', 'IK_R', 'IA_P', 'IA_R', 'Confidence', 'Type']].copy()
        display_df = display_df.sort_index(ascending=False)
        display_df.insert(len(display_df.columns), "Del", False)
        
        edited = st.data_editor(
            display_df,
            column_config={
                "Del": st.column_config.CheckboxColumn("🗑️", width="small"),
                "Date": st.column_config.DateColumn("Date", format="DD.MM"),
                "H": st.column_config.TextColumn("Dom", width="small"),
                "A": st.column_config.TextColumn("Ext", width="small"),
                "Win": st.column_config.TextColumn("Vainq", width="small"),
                
                "IK_P": st.column_config.TextColumn("Moi", width="small"),
                "IK_R": st.column_config.TextColumn("Res", width="small", disabled=True), # Read Only
                
                "IA_P": st.column_config.TextColumn("IA", width="small"),
                "IA_R": st.column_config.TextColumn("Res", width="small", disabled=True), # Read Only
                
                "Confidence": st.column_config.TextColumn("%", width="small"),
                "Type": st.column_config.TextColumn("Typ", width="small"),
            },
            height=600,
            hide_index=True,
            use_container_width=True
        )
        
        c1, c2 = st.columns(2)
        if c1.button("Supprimer la sélection"):
            to_del_idx = edited[edited.Del == True].index
            if not to_del_idx.empty:
                # Reload clean, drop and save
                orig = pd.read_csv(HISTORY_FILE)
                orig.drop(to_del_idx, inplace=True)
                orig.to_csv(HISTORY_FILE, index=False)
                st.success("Supprimé !"); time.sleep(0.5); st.rerun()
        if c2.button("Supprimer les doublons"):
             orig = pd.read_csv(HISTORY_FILE)
             orig.drop_duplicates(subset=['Date', 'Home', 'Away'], keep='last', inplace=True)
             orig.to_csv(HISTORY_FILE, index=False)
             st.success("Doublons supprimés"); time.sleep(0.5); st.rerun()

# ==============================================================================
# TAB 3 : ADMIN
# ==============================================================================
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"Modèle : {get_last_mod(MODEL_FILE)}")
        if st.button("Force Update", type="primary", use_container_width=True):
            with st.status("Mise à jour...") as s:
                run_script('src/data_nba.py', "Data", s)
                run_script('src/features_nba.py', "Stats", s)
                run_script('src/verify_bets.py', "Verif", s)
                st.session_state['schedule_data'] = {} 
                load_resources.clear()
                s.update(label="Terminé", state="complete")
                st.rerun()
    with c2:
        st.info(f"Données : {get_last_mod(GAMES_FILE)}")
        if st.button("Entraînement Hebdo", use_container_width=True):
            with st.status("Training...") as s:
                succ, msg, acc = train_nba.train_model()
                if succ:
                    run_script('src/features_nba.py', "Stats", s)
                    load_resources.clear()
                    s.update(label=f"Succès ({acc:.1%})", state="complete")
                else: s.error(msg)