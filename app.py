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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NBA | AGENT PREDiKTOR", page_icon="🏀", layout="wide")

# --- CSS COMPLET ---
st.markdown("""
<style>
    /* 1. HEADER & NAVIGATION FIXES */
    /* Force le header natif à rester en haut */
    header[data-testid="stHeader"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 99999 !important;
        background-color: #0e1117 !important;
        opacity: 1 !important;
        height: 3.5rem !important;
    }
    /* Force la barre de navigation (Tabs) à coller au header */
    div[data-testid="stTabs"] {
        position: sticky !important;
        top: 3.5rem !important; /* Hauteur du header */
        z-index: 99990 !important;
        background-color: #0e1117 !important;
        padding-top: 0px !important;
        border-bottom: 1px solid #333;
    }
    /* Pousse le contenu vers le bas pour ne pas être caché par le header */
    .main .block-container {
        padding-top: 7rem !important; 
    }

    /* 2. CARDS STYLING */
    .match-card-container {
        border: 1px solid #333;
        border-radius: 8px;
        background-color: #161920;
        margin-bottom: 10px;
        overflow: hidden;
    }
    
    /* Zone Prono avec FOND GRIS DISTINCT */
    .prono-container {
        background-color: #262730; 
        border-radius: 6px;
        padding: 8px;
        margin: 5px;
        text-align: center;
        border: 1px solid #3e3e3e;
    }
    
    .team-name { font-weight: bold; font-size: 1.0em; margin:0; }
    .team-meta { font-size: 0.75em; color: #aaa; }
    .prono-label { font-size: 0.7em; color: #bbb; text-transform: uppercase; letter-spacing: 1px; }
    .prono-val { font-size: 1.4em; font-weight: 900; color: #fff; line-height: 1.2; }
    .prono-conf { font-size: 0.8em; color: #00d4ff; font-weight: bold; }
    
    /* Bouton Modifier Discret */
    div[data-testid="stButton"] button {
        border-radius: 4px;
    }
    
    /* 3. TABLEAU RESULTATS HTML (Zebra) */
    .res-table {
        width: 100%;
        border-collapse: collapse;
        font-family: sans-serif;
        font-size: 0.9em;
    }
    .res-table th { text-align: left; color: #888; font-size: 0.8em; padding: 5px; border-bottom: 1px solid #444; }
    .res-table td { padding: 8px 5px; border-bottom: 1px solid #333; color: #eee; }
    /* Zebra Striping */
    .res-table tr:nth-child(even) { background-color: #1f2129; }
    .res-table tr:nth-child(odd) { background-color: #16181e; }
    
    .badge-win { color: #4ade80; font-weight: bold; }
    .badge-loss { color: #f87171; font-weight: bold; }
    .badge-neutral { color: #666; }

    /* 4. MOBILE OPTIMIZATIONS */
    @media (max-width: 640px) {
        /* Layout spécifique mobile pour les cards */
        .mobile-teams-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #333;
        }
        .mobile-team { text-align: center; width: 45%; }
        
        /* Cacher la vue desktop sur mobile et inversement */
        .desktop-view { display: none !important; }
        .mobile-view { display: block !important; }
        
        /* Boutons */
        .stButton button { width: 100%; margin-top: 5px; }
    }
    
    @media (min-width: 641px) {
        .mobile-view { display: none !important; }
        .desktop-view { display: flex !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SESSIONS ---
if 'schedule_data' not in st.session_state: st.session_state['schedule_data'] = {}
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

            res[tid] = {'rec': row['Record'], 'strk': streak_short, 'rank': row['PlayoffRank']}
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
    except subprocess.CalledProcessError:
        container.error(f"Erreur {desc}")
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
        st.session_state['edit_modes'][f"{h_name} vs {a_name}"] = False
    except: pass

def get_last_mod(filepath):
    if os.path.exists(filepath): return datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%d/%m %H:%M')
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

# --- HEADER (Visible) ---
c_head1, c_head2 = st.columns([1, 8])
with c_head1:
    if os.path.exists(APP_LOGO): st.image(APP_LOGO, width=60)
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
            
            matches_to_display = []
            seen = []
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
                    
                    mid = f"{h_name}vs{a_name}"
                    if mid in seen: continue
                    seen.append(mid)

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
                        matches_to_display.append({'h': h_name, 'a': a_name, 'hid': h_id, 'aid': a_id, 'prob': prob, 'u': user_bet_val, 'mid': mid, 'd': date_key})

            # --- RENDER CARDS ---
            if matches_to_display:
                cols = st.columns(2)
                for i, m in enumerate(matches_to_display):
                    with cols[i % 2]:
                        with st.container():
                            # CSS Class Wrapper
                            st.markdown('<div class="match-card-container">', unsafe_allow_html=True)
                            
                            inf_h = STANDINGS_DB.get(m['hid'], {'rec': '', 'strk': '', 'rank': ''})
                            inf_a = STANDINGS_DB.get(m['aid'], {'rec': '', 'strk': '', 'rank': ''})
                            c_sh = "#4ade80" if 'W' in inf_h['strk'] else "#f87171"
                            c_sa = "#4ade80" if 'W' in inf_a['strk'] else "#f87171"
                            
                            is_h_win = m['prob'] > 0.5
                            ia_conf = m['prob']*100 if is_h_win else (1-m['prob'])*100
                            ia_code = TEAMS_DB.get(m['hid'] if is_h_win else m['aid'], {}).get('code', 'IA')
                            
                            # --- MOBILE LAYOUT (Team - Team sur ligne 1) ---
                            st.markdown(f"""
                            <div class='mobile-view'>
                                <div class='mobile-teams-row'>
                                    <div class='mobile-team'>
                                        <img src='https://cdn.nba.com/logos/nba/{m['hid']}/global/L/logo.svg' width='35'><br>
                                        <span class='team-name'>{TEAMS_DB.get(m['hid'],{}).get('code', 'H')}</span><br>
                                        <span class='team-meta'>#{inf_h['rank']}</span>
                                    </div>
                                    <div style='font-weight:bold; color:#666;'>VS</div>
                                    <div class='mobile-team'>
                                        <img src='https://cdn.nba.com/logos/nba/{m['aid']}/global/L/logo.svg' width='35'><br>
                                        <span class='team-name'>{TEAMS_DB.get(m['aid'],{}).get('code', 'A')}</span><br>
                                        <span class='team-meta'>#{inf_a['rank']}</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # --- DESKTOP LAYOUT (3 colonnes) ---
                            st.markdown("<div class='desktop-view' style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
                            c1, c2, c3 = st.columns([3, 4, 3])
                            with c1:
                                st.markdown(f"""
                                <div class='team-col'>
                                    <img src='https://cdn.nba.com/logos/nba/{m['hid']}/global/L/logo.svg' width='45'>
                                    <div class='team-name'>{TEAMS_DB.get(m['hid'],{}).get('code', 'H')}</div>
                                    <div class='team-meta'>#{inf_h['rank']} ({inf_h['rec']})<br><span style='color:{c_sh}; font-weight:bold;'>{inf_h['strk']}</span></div>
                                </div>""", unsafe_allow_html=True)
                            with c2: st.empty() # Spacer desktop (la zone prono est commune dessous)
                            with c3:
                                st.markdown(f"""
                                <div class='team-col'>
                                    <img src='https://cdn.nba.com/logos/nba/{m['aid']}/global/L/logo.svg' width='45'>
                                    <div class='team-name'>{TEAMS_DB.get(m['aid'],{}).get('code', 'A')}</div>
                                    <div class='team-meta'>#{inf_a['rank']} ({inf_a['rec']})<br><span style='color:{c_sa}; font-weight:bold;'>{inf_a['strk']}</span></div>
                                </div>""", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # --- ZONE PRONO (Commune Mobile/Desktop) ---
                            # On utilise des colonnes Streamlit pour le centrage du bouton
                            pc1, pc2, pc3 = st.columns([1, 4, 1])
                            with pc2:
                                st.markdown(f"""
                                <div class='prono-container'>
                                    <div class='prono-label'>IA PRONO</div>
                                    <div class='prono-val'>{ia_code}</div>
                                    <div class='prono-conf'>{ia_conf:.0f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # USER INTERACTION
                                has_voted = (m['u'] is not None and m['u'] != "")
                                is_editing = st.session_state['edit_modes'].get(m['mid'], False)
                                
                                if has_voted and not is_editing:
                                    u_code = TEAMS_DB.get(next((k for k,v in TEAMS_DB.items() if v['full'] == m['u']),0), {}).get('code', m['u'])
                                    st.markdown(f"""
                                        <div style='text-align:center; font-size:0.8em; margin-top:5px;'>
                                            Choix IK: <b>{u_code}</b>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    # Bouton Modifier petit et centré
                                    if st.button("Modifier", key=f"btn_mod_{m['mid']}", use_container_width=True):
                                        st.session_state['edit_modes'][m['mid']] = True
                                        st.rerun()
                                else:
                                    # Boutons Vote
                                    b1, b2 = st.columns(2)
                                    ch = TEAMS_DB.get(m['hid'], {}).get('code', 'H')
                                    ca = TEAMS_DB.get(m['aid'], {}).get('code', 'A')
                                    if b1.button(ch, key=f"bh_{m['mid']}", use_container_width=True):
                                        save_user_vote(m['d'], m['h'], m['a'], m['h'])
                                        st.rerun()
                                    if b2.button(ca, key=f"ba_{m['mid']}", use_container_width=True):
                                        save_user_vote(m['d'], m['h'], m['a'], m['a'])
                                        st.rerun()

                            st.markdown('</div>', unsafe_allow_html=True) # End Card Container

    elif st.session_state['schedule_data'] == {}:
        st.info("Aucun match.")

    # 4. RESULTATS (Tableau HTML Zebra)
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        finished = hist[hist['Result'].isin(['GAGNE', 'PERDU'])].copy()
        
        if not finished.empty:
            st.write("")
            st.markdown("#### 🏁 Derniers Résultats")
            
            # Centrage du tableau résultats
            c_res_main, _ = st.columns([1, 1]) # Limit width
            
            with c_res_main:
                dates = sorted(finished['Date'].unique(), reverse=True)[:2]
                first_open = True
                
                for d in dates:
                    day_rows = finished[finished['Date'] == d]
                    ia_wins = len(day_rows[day_rows['Result'] == 'GAGNE'])
                    user_wins = 0
                    if 'User_Result' in day_rows.columns:
                        user_wins = len(day_rows[day_rows['User_Result'] == 'GAGNE'])
                    
                    try: d_fmt = datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m')
                    except: d_fmt = d
                    
                    with st.expander(f"📅 {d_fmt} | IA: {ia_wins}/{len(day_rows)} | IK: {user_wins}/{len(day_rows)}", expanded=first_open):
                        first_open = False # Seul le premier est ouvert
                        
                        # Construction HTML Table
                        html_table = "<table class='res-table'><tr><th>MATCH</th><th>WIN</th><th>IA</th><th>IK</th></tr>"
                        
                        for _, r in day_rows.iterrows():
                            match_str = f"{get_short_code(r['Home'])}-{get_short_code(r['Away'])}"
                            win_str = get_short_code(r['Real_Winner']) if pd.notna(r['Real_Winner']) else "?"
                            
                            # Badges
                            ia_cls = "badge-win" if r['Result'] == 'GAGNE' else "badge-loss"
                            ia_txt = "OK" if r['Result'] == 'GAGNE' else "KO"
                            
                            ik_txt = "-"
                            ik_cls = "badge-neutral"
                            if 'User_Result' in r and pd.notna(r['User_Result']):
                                ik_txt = "OK" if r['User_Result'] == 'GAGNE' else "KO" if r['User_Result'] == 'PERDU' else "-"
                                ik_cls = "badge-win" if r['User_Result'] == 'GAGNE' else "badge-loss" if r['User_Result'] == 'PERDU' else "badge-neutral"

                            html_table += f"<tr><td>{match_str}</td><td>{win_str}</td><td class='{ia_cls}'>{ia_txt}</td><td class='{ik_cls}'>{ik_txt}</td></tr>"
                        
                        html_table += "</table>"
                        st.markdown(html_table, unsafe_allow_html=True)

# ==============================================================================
# TAB 2 : STATS
# ==============================================================================
with tab2:
    # Centrage du tableau
    _, c_tab_center, _ = st.columns([1, 10, 1])
    
    with c_tab_center:
        if os.path.exists(HISTORY_FILE):
            df_hist = pd.read_csv(HISTORY_FILE)
            df_hist = df_hist.fillna("")
            df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
            
            # FUSION LOGIC
            def merge_prono_res(prono, res):
                if not prono or prono == "": return "..."
                p_code = get_short_code(get_clean_name(prono))
                if not res or res not in ['GAGNE', 'PERDU']: return f"{p_code}"
                icon = "✅" if res == 'GAGNE' else "❌"
                return f"{icon} {p_code}"

            df_hist['Home'] = df_hist['Home'].apply(lambda x: get_short_code(get_clean_name(x)))
            df_hist['Away'] = df_hist['Away'].apply(lambda x: get_short_code(get_clean_name(x)))
            df_hist['Winner'] = df_hist['Real_Winner'].apply(lambda x: get_short_code(get_clean_name(x)) if x not in ["En attente...", ""] else "...")
            
            df_hist['Prono IA'] = df_hist.apply(lambda x: merge_prono_res(x['Predicted_Winner'], x['Result']), axis=1)
            
            if 'User_Prediction' not in df_hist.columns: df_hist['User_Prediction'] = ""
            if 'User_Result' not in df_hist.columns: df_hist['User_Result'] = ""
            df_hist['Prono IK'] = df_hist.apply(lambda x: merge_prono_res(x['User_Prediction'], x['User_Result']), axis=1)

            df_hist['Trust'] = df_hist['Confidence']

            cols_order = ['Date', 'Home', 'Away', 'Winner', 'Prono IK', 'Prono IA', 'Trust', 'Type']
            display_df = df_hist[cols_order].copy()
            display_df = display_df.sort_index(ascending=False)
            display_df.insert(len(display_df.columns), "Del", False)
            
            edited = st.data_editor(
                display_df,
                column_config={
                    "Del": st.column_config.CheckboxColumn("🗑️", width="small"),
                    "Date": st.column_config.DateColumn("Date", format="DD.MM"),
                    "Home": st.column_config.TextColumn("Home", width="small"),
                    "Away": st.column_config.TextColumn("Away", width="small"),
                    "Winner": st.column_config.TextColumn("Winner", width="small"),
                    "Prono IK": st.column_config.TextColumn("Prono IK", width="small"), # Réduit
                    "Prono IA": st.column_config.TextColumn("Prono IA", width="small"), # Réduit
                    "Trust": st.column_config.TextColumn("Trust", width="small"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("Supprimer la sélection"):
                to_del_idx = edited[edited.Del == True].index
                if not to_del_idx.empty:
                    orig = pd.read_csv(HISTORY_FILE)
                    orig.drop(to_del_idx, inplace=True)
                    orig.to_csv(HISTORY_FILE, index=False)
                    st.success("Supprimé"); time.sleep(0.5); st.rerun()

# ==============================================================================
# TAB 3 : ADMIN
# ==============================================================================
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"Modèle : {get_last_mod(MODEL_FILE)}")
        if st.button("Force Update", type="primary"):
            with st.status("Update...") as s:
                run_script('src/data_nba.py', "Data", s)
                run_script('src/features_nba.py', "Stats", s)
                run_script('src/verify_bets.py', "Verif", s)
                st.session_state['schedule_data'] = {} 
                load_resources.clear()
                s.update(label="Terminé", state="complete")
                st.rerun()
    with c2:
        st.info(f"Données : {get_last_mod(GAMES_FILE)}")
        if st.button("Entraînement"):
            with st.status("Training...") as s:
                succ, msg, acc = train_nba.train_model()
                if succ:
                    run_script('src/features_nba.py', "Stats", s)
                    load_resources.clear()
                    s.update(label=f"Succès ({acc:.1%})", state="complete")
                else: s.error(msg)
                
    st.markdown("---")
    st.subheader("🔮 Ajout Manuel")
    
    # Restauration de l'ajout manuel
    cm1, cm2, cm3 = st.columns(3)
    team_names = [f"{v['code']} - {v['full']}" for k,v in TEAMS_DB.items()]
    hm = cm1.selectbox("Home", team_names, index=None)
    aw = cm2.selectbox("Away", team_names, index=None)
    dt = cm3.date_input("Date", value=datetime.now())
    
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
                st.session_state['schedule_data'] = {} 
                st.rerun()