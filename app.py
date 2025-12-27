import streamlit as st
import pandas as pd
import xgboost as xgb
import altair as alt
from datetime import datetime, timedelta
import os
import sys
import subprocess
import time
import requests
import json
from nba_api.stats.static import teams
from nba_api.stats.endpoints import scoreboardv2, leaguestandingsv3, leaguegamefinder
from src import train_nba

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NBA | AGENT PREDiKTOR", page_icon="🏀", layout="wide")

# --- SUPABASE CREDENTIALS ---
SUPABASE_URL = "https://meqvmpqyizffzlvomqbb.supabase.co/" 
SUPABASE_KEY = "sb_publishable_bSPoeHBKUrxsEwn0ZI5cdA_iAAK3wza"

# --- CSS (DESIGN V9 UNIFIED) ---
st.markdown("""
<style>
    .stApp { background-color: #262730 !important; }
    header[data-testid="stHeader"] { background-color: #262730 !important; }
    div[data-testid="stTabs"] {
        position: sticky;
        top: 2.8rem;
        background-color: #262730 !important;
        z-index: 999;
        padding-top: 10px;
        margin-top: 0px;
        border-bottom: 1px solid #444;
    }
    .unified-card {
        background-color: #262730;
        border: 1px solid #444;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .stat-box { text-align: center; padding: 10px; border-right: 1px solid rgba(255,255,255,0.1); }
    .stat-box:last-child { border-right: none; }
    .stat-label { font-size: 0.8em; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .stat-val { font-size: 2em; font-weight: 900; color: #fff; line-height: 1.1; }
    .stat-sub { font-size: 0.8em; font-weight: bold; }
    .color-ai { color: #00d4ff !important; }
    .color-ik { color: #4ade80 !important; }
    .color-bad { color: #f87171 !important; }
    .card-header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px; }
    .team-box { flex: 1; text-align: center; display: flex; flex-direction: column; align-items: center; }
    .vs-text { font-weight: bold; color: #666; font-size: 0.8em; padding: 0 10px; }
    .t-code { font-weight: bold; font-size: 1.3em; margin-top: 5px; line-height: 1; color: #fff; }
    .t-meta { font-size: 0.75em; color: #bbb; margin-top: 4px; }
    .score-val { font-size: 1.8em; font-weight: 900; color: #fff; }
    .score-win { color: #4ade80; }
    .score-loss { color: #aaa; opacity: 0.6; }
    .status-final { font-size: 0.7em; color: #f87171; font-weight: bold; letter-spacing: 1px; border: 1px solid #f87171; padding: 2px 6px; border-radius: 4px; }
    .prono-section { display: flex; flex-direction: column; gap: 8px; align-items: center; width: 100%; }
    .prono-row { display: flex; align-items: center; justify-content: center; gap: 15px; font-size: 0.9em; width: 100%; }
    .p-lbl { color: #888; font-weight: bold; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; }
    .p-val { color: #fff; font-weight: 900; font-size: 1.3em; }
    .p-conf { color: #00d4ff; font-size: 0.85em; font-weight: bold; }
    .res-badge-win { background-color: rgba(74, 222, 128, 0.2); color: #4ade80; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; border: 1px solid #4ade80; }
    .res-badge-loss { background-color: rgba(248, 113, 113, 0.2); color: #f87171; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; border: 1px solid #f87171; }
    .user-choice-row { margin-top: 5px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05); width: 100%; text-align: center; }
    .reason-text { font-size: 0.8em; color: #aaa; font-style: italic; margin-top: 2px; }
    .action-container { margin-top: 5px; margin-bottom: 15px; text-align: center; }
    .link-btn button { background: transparent !important; border: none !important; color: #666 !important; text-decoration: underline !important; padding: 0 !important; font-size: 0.75em !important; height: auto !important; margin-top: 2px !important; }
    .link-btn button:hover { color: #fff !important; }
    div[data-baseweb="select"] > div { background-color: #1e2026 !important; border-color: #444 !important; font-size: 0.8em !important; }
    .res-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    .res-table th { text-align: left; color: #888; border-bottom: 1px solid #444; padding: 5px; }
    .res-table td { border-bottom: 1px solid #333; padding: 8px 5px; color: #ddd; }
    .res-table tr:nth-child(even) { background-color: #2d2f38; }
    .res-table tr:nth-child(odd) { background-color: #262730; }
    @media (max-width: 640px) {
        .t-code { font-size: 1.1em; }
        .stButton button { width: 100%; }
        .unified-card { padding: 10px; }
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

REASONS_LIST = [
    "Intuition / Feeling", "Blessure / Effectif", "Série / Forme du moment",
    "Domicile / Extérieur", "Analyse Stats", "Revanche / Rivalité",
    "Back-to-Back (Fatigue)", "Cote / Value Bet", "Suivi de l'IA", "Contre l'IA"
]

# --- 3. FONCTIONS ---

# --- SUPABASE FUNCTIONS ---
@st.cache_data(ttl=60)
def load_history_from_supabase():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{SUPABASE_URL}/rest/v1/bets_history?select=*"
    
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            if not data: return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Fix KeyError: Si 'type' n'est pas dans le cloud, on le crée
            if 'type' not in df.columns:
                df['type'] = 'Auto'

            rename_map = {
                'game_date': 'Date',
                'home_team': 'Home',
                'away_team': 'Away',
                'predicted_winner': 'Predicted_Winner',
                'confidence': 'Confidence',
                'result_ia': 'Result',
                'real_winner': 'Real_Winner',
                'user_prediction': 'User_Prediction',
                'user_result': 'User_Result',
                'user_reason': 'User_Reason',
                'type': 'Type'
            }
            df = df.rename(columns=rename_map)
            return df
        else:
            st.error(f"Erreur Supabase: {r.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur Connection: {e}")
        return pd.DataFrame()

def save_user_vote_cloud(date_str, h_name, a_name, user_choice, reason, match_key):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    url = f"{SUPABASE_URL}/rest/v1/bets_history?game_date=eq.{date_str}&home_team=eq.{h_name}&away_team=eq.{a_name}"
    
    payload = {
        "user_prediction": user_choice,
        "user_reason": reason
    }
    
    try:
        r = requests.patch(url, headers=headers, json=payload)
        if r.status_code in [200, 204]:
            st.toast("Vote enregistré dans le Cloud ☁️", icon="✅")
            st.session_state['edit_modes'][match_key] = False
            load_history_from_supabase.clear()
        else:
            st.error(f"Erreur sauvegarde: {r.text}")
    except Exception as e:
        st.error(f"Erreur: {e}")

# --- FONCTIONS LOCALES ---

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

def save_bet_auto_local(date, h_name, a_name, w_name, conf):
    if not os.path.exists(HISTORY_FILE): 
        with open(HISTORY_FILE, 'w') as f: f.write("Date,Home,Away,Predicted_Winner,Confidence,Type,Result,Real_Winner,User_Prediction,User_Result,User_Reason\n")
    try:
        df = pd.read_csv(HISTORY_FILE)
        if not df[(df['Date'] == date) & (df['Home'] == h_name) & (df['Away'] == a_name)].empty: return
    except: pass
    with open(HISTORY_FILE, 'a') as f:
        f.write(f"\n{date},{h_name},{a_name},{w_name},{conf:.1f}%,Auto,,,,")

def get_last_mod(filepath):
    if os.path.exists(filepath): return datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%d/%m %H:%M')
    return "N/A"

def clean_id_hard(val):
    try: return str(int(float(val))).lstrip('0')
    except: return str(val).lstrip('0')

# --- SCANNER V9.21 ---
def scan_schedule(days_to_check=7):
    found_days = {}
    check_date = datetime.now() - timedelta(days=1) 
    
    # CHARGEMENT CLOUD
    hist_data = load_history_from_supabase()
    
    count_found = 0
    for _ in range(days_to_check):
        str_date = check_date.strftime('%Y-%m-%d')
        finder_date = check_date.strftime('%m/%d/%Y')
        day_games_list = []
        try:
            board = scoreboardv2.ScoreboardV2(game_date=str_date)
            raw = board.game_header.get_data_frame()
            clean = raw.dropna(subset=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
            
            if not clean.empty:
                finder = leaguegamefinder.LeagueGameFinder(
                    date_from_nullable=finder_date,
                    date_to_nullable=finder_date,
                    league_id_nullable='00'
                )
                results = finder.get_data_frames()[0]
                
                score_map = {}
                if not results.empty:
                    for _, r in results.iterrows():
                        try:
                            gid = clean_id_hard(r['GAME_ID'])
                            tid = clean_id_hard(r['TEAM_ID'])
                            pts = int(r['PTS'])
                            key = f"{gid}_{tid}"
                            score_map[key] = pts
                        except: continue
                
                def get_score(row, is_home):
                    try:
                        g_id = clean_id_hard(row['GAME_ID'])
                        t_id = clean_id_hard(row['HOME_TEAM_ID']) if is_home else clean_id_hard(row['VISITOR_TEAM_ID'])
                        key = f"{g_id}_{t_id}"
                        return score_map.get(key, None)
                    except: return None

                clean['PTS_HOME'] = clean.apply(lambda row: get_score(row, True), axis=1)
                clean['PTS_AWAY'] = clean.apply(lambda row: get_score(row, False), axis=1)
                
                def force_status(row):
                    if pd.notna(row['PTS_HOME']) and pd.notna(row['PTS_AWAY']):
                        return 3
                    return row.get('GAME_STATUS_ID', 1)

                clean['GAME_STATUS_ID'] = clean.apply(force_status, axis=1)
                day_games_list.append(clean)
                
        except: pass
        
        if not hist_data.empty:
            try:
                manual_today = hist_data[(hist_data['Date'] == str_date)].copy()
                if not manual_today.empty: 
                    manual_today['GAME_STATUS_ID'] = 3 
                    manual_today['PTS_HOME'] = 0
                    manual_today['PTS_AWAY'] = 0
                    mask_unfinished = manual_today['Result'].isna() | (manual_today['Result'] == "")
                    manual_today.loc[mask_unfinished, 'GAME_STATUS_ID'] = 1
                    day_games_list.append(manual_today)
            except: pass
            
        if day_games_list:
            found_days[str_date] = day_games_list
            count_found += 1
            
        if count_found >= 2: break 
        check_date += timedelta(days=1)
        
    return found_days

# --- INIT ---
model, df_stats = load_resources()

# --- HEADER ---
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
        with st.spinner("Chargement Cloud..."):
            st.session_state['schedule_data'] = scan_schedule()

    schedule = st.session_state.get('schedule_data', {})
    
    # ICI : On lit le DF Cloud
    hist_df = load_history_from_supabase()

    if schedule:
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
                    
                    status_id = row.get('GAME_STATUS_ID', 1) 
                    pts_h = row.get('PTS_HOME', None)
                    pts_a = row.get('PTS_AWAY', None)
                    
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
                    user_reason_val = None
                    real_winner_hist = None
                    
                    if not hist_df.empty:
                        existing_bet = hist_df[(hist_df['Date'] == date_key) & (hist_df['Home'] == h_name) & (hist_df['Away'] == a_name)]
                    
                    if not existing_bet.empty:
                        saved_row = existing_bet.iloc[0]
                        winner = saved_row['Predicted_Winner']
                        real_winner_hist = saved_row.get('Real_Winner', None)
                        
                        if 'User_Prediction' in saved_row and pd.notna(saved_row['User_Prediction']):
                            user_bet_val = saved_row['User_Prediction']
                        if 'User_Reason' in saved_row and pd.notna(saved_row['User_Reason']):
                            user_reason_val = saved_row['User_Reason']
                        try:
                            conf_str = str(saved_row['Confidence']).replace('%', '')
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
                                save_bet_auto_local(date_key, h_name, a_name, w, c) 
                                st.rerun()
                    
                    if prob is not None and h_id != 0:
                        matches_to_display.append({
                            'h': h_name, 'a': a_name, 'hid': h_id, 'aid': a_id, 
                            'prob': prob, 'u': user_bet_val, 'reason': user_reason_val, 'mid': mid, 'd': date_key,
                            'status': status_id, 'pts_h': pts_h, 'pts_a': pts_a,
                            'real_winner': real_winner_hist 
                        })

            # --- RENDER CARDS ---
            if matches_to_display:
                cols = st.columns(2)
                for i, m in enumerate(matches_to_display):
                    with cols[i % 2]:
                        with st.container():
                            try:
                                score_h = int(float(m['pts_h'])) if pd.notna(m['pts_h']) else 0
                                score_a = int(float(m['pts_a'])) if pd.notna(m['pts_a']) else 0
                            except: score_h, score_a = 0, 0
                            
                            has_scores = (score_h > 0 and score_a > 0)
                            has_winner_hist = (m['real_winner'] is not None and str(m['real_winner']) not in ['nan', 'None', '', 'En attente...'])
                            
                            is_finished = (m['status'] == 3) or has_scores or has_winner_hist
                            
                            inf_h = STANDINGS_DB.get(m['hid'], {'rec': '', 'strk': '', 'rank': ''})
                            inf_a = STANDINGS_DB.get(m['aid'], {'rec': '', 'strk': '', 'rank': ''})
                            c_sh = "#4ade80" if 'W' in inf_h['strk'] else "#f87171"
                            c_sa = "#4ade80" if 'W' in inf_a['strk'] else "#f87171"
                            is_h_win = m['prob'] > 0.5
                            ia_conf = m['prob']*100 if is_h_win else (1-m['prob'])*100
                            ia_code = TEAMS_DB.get(m['hid'] if is_h_win else m['aid'], {}).get('code', 'IA')
                            
                            has_voted = (m['u'] is not None and m['u'] != "")
                            is_editing = st.session_state['edit_modes'].get(m['mid'], False)
                            
                            # HEADER
                            score_vs_block = "<div class='vs-text'>VS</div>"
                            if is_finished:
                                h_cls = "score-win" if score_h > score_a else "score-loss"
                                a_cls = "score-win" if score_a > score_h else "score-loss"
                                score_vs_block = f"<div class='status-final'>FINAL</div>"
                                
                                d_score_h = score_h if has_scores else "?"
                                d_score_a = score_a if has_scores else "?"
                                
                                html_teams = f"""
                                <div class='card-header'>
                                    <div class='team-box'>
                                        <img src='https://cdn.nba.com/logos/nba/{m['hid']}/global/L/logo.svg' width='40'>
                                        <span class='t-code'>{TEAMS_DB.get(m['hid'],{}).get('code', 'H')}</span>
                                        <span class='score-val {h_cls}'>{d_score_h}</span>
                                    </div>
                                    {score_vs_block}
                                    <div class='team-box'>
                                        <img src='https://cdn.nba.com/logos/nba/{m['aid']}/global/L/logo.svg' width='40'>
                                        <span class='t-code'>{TEAMS_DB.get(m['aid'],{}).get('code', 'A')}</span>
                                        <span class='score-val {a_cls}'>{d_score_a}</span>
                                    </div>
                                </div>
                                """
                            else:
                                html_teams = f"""
                                <div class='card-header'>
                                    <div class='team-box'>
                                        <img src='https://cdn.nba.com/logos/nba/{m['hid']}/global/L/logo.svg' width='40'>
                                        <span class='t-code'>{TEAMS_DB.get(m['hid'],{}).get('code', 'H')}</span>
                                        <span class='t-meta'>#{inf_h['rank']} ({inf_h['rec']}) <b style='color:{c_sh}'>{inf_h['strk']}</b></span>
                                    </div>
                                    {score_vs_block}
                                    <div class='team-box'>
                                        <img src='https://cdn.nba.com/logos/nba/{m['aid']}/global/L/logo.svg' width='40'>
                                        <span class='t-code'>{TEAMS_DB.get(m['aid'],{}).get('code', 'A')}</span>
                                        <span class='t-meta'>#{inf_a['rank']} ({inf_a['rec']}) <b style='color:{c_sa}'>{inf_a['strk']}</b></span>
                                    </div>
                                </div>
                                """
                            
                            # PRONO
                            if is_finished:
                                real_winner_name = None
                                if has_scores:
                                    real_winner_name = m['h'] if score_h > score_a else m['a']
                                elif has_winner_hist:
                                    real_winner_name = m['real_winner']
                                
                                if real_winner_name:
                                    ia_pred_name = m['h'] if is_h_win else m['a']
                                    ia_res = "res-badge-win" if ia_pred_name == real_winner_name else "res-badge-loss"
                                    ia_txt = "GAGNÉ" if ia_pred_name == real_winner_name else "PERDU"
                                    
                                    html_ia = f"<div class='prono-row'><span class='p-lbl'>IA</span><span class='p-val'>{ia_code}</span><span class='{ia_res}'>{ia_txt}</span></div>"
                                    
                                    html_user = ""
                                    if has_voted:
                                        u_code = TEAMS_DB.get(next((k for k,v in TEAMS_DB.items() if v['full'] == m['u']),0), {}).get('code', m['u'])
                                        u_win = (m['u'] == real_winner_name)
                                        u_res = "res-badge-win" if u_win else "res-badge-loss"
                                        u_txt = "GAGNÉ" if u_win else "PERDU"
                                        html_user = f"<div class='user-choice-row'><div class='prono-row' style='justify-content:center;'><span class='p-lbl'>IK</span><span class='p-val'>{u_code}</span><span class='{u_res}'>{u_txt}</span></div></div>"
                                    else:
                                        html_user = f"<div class='user-choice-row'><div class='prono-row' style='justify-content:center;'><span class='p-lbl'>IK</span><span style='color:#666;'>-</span></div></div>"
                                else:
                                    html_ia = f"<div class='prono-row'><span class='p-lbl'>IA</span><span class='p-val'>{ia_code}</span></div>"
                                    html_user = ""

                            else:
                                html_ia = f"<div class='prono-row'><span class='p-lbl'>IA</span><span class='p-val'>{ia_code}</span><span class='p-conf'>{ia_conf:.0f}%</span></div>"
                                html_user = ""
                                if has_voted and not is_editing:
                                    u_code = TEAMS_DB.get(next((k for k,v in TEAMS_DB.items() if v['full'] == m['u']),0), {}).get('code', m['u'])
                                    reason_disp = f"<div class='reason-text'>({m['reason']})</div>" if m['reason'] else ""
                                    html_user = f"<div class='user-choice-row'><div class='prono-row' style='justify-content:center;'><span class='p-lbl'>IK</span><span class='p-val'>{u_code}</span></div>{reason_disp}</div>"
                            
                            st.markdown(f"<div class='unified-card'>{html_teams}<div class='prono-section'>{html_ia}{html_user}</div></div>", unsafe_allow_html=True)
                            
                            if not is_finished:
                                st.markdown('<div class="action-container">', unsafe_allow_html=True)
                                if has_voted and not is_editing:
                                    st.markdown('<div class="link-btn">', unsafe_allow_html=True)
                                    if st.button("Modifier", key=f"btn_mod_{m['mid']}", width="stretch"):
                                        st.session_state['edit_modes'][m['mid']] = True
                                        st.rerun()
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    reason_choice = st.selectbox("Justification", REASONS_LIST, key=f"reason_{m['mid']}", label_visibility="collapsed")
                                    b1, b2 = st.columns(2)
                                    ch = TEAMS_DB.get(m['hid'], {}).get('code', 'H')
                                    ca = TEAMS_DB.get(m['aid'], {}).get('code', 'A')
                                    if b1.button(ch, key=f"bh_{m['mid']}", width="stretch"):
                                        save_user_vote_cloud(m['d'], m['h'], m['a'], m['h'], reason_choice, m['mid'])
                                        st.rerun()
                                    if b2.button(ca, key=f"ba_{m['mid']}", width="stretch"):
                                        save_user_vote_cloud(m['d'], m['h'], m['a'], m['a'], reason_choice, m['mid'])
                                        st.rerun()
                                st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state['schedule_data'] == {}:
        st.info("Aucun match.")

    # 4. RESULTATS
    if not hist_df.empty:
        finished = hist_df[hist_df['Result'].isin(['GAGNE', 'PERDU'])].copy()
        
        if not finished.empty:
            st.write("")
            st.markdown("#### 🏁 Derniers Résultats")
            c_res_main, _ = st.columns([1, 1]) 
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
                        first_open = False
                        html_table = "<table class='res-table'><tr><th>MATCH</th><th>WIN</th><th>IA</th><th>IK</th></tr>"
                        for _, r in day_rows.iterrows():
                            match_str = f"{get_short_code(r['Home'])}-{get_short_code(r['Away'])}"
                            win_str = get_short_code(r['Real_Winner']) if pd.notna(r['Real_Winner']) else "?"
                            col_ia = "#4ade80" if r['Result'] == 'GAGNE' else "#f87171"
                            txt_ia = "OK" if r['Result'] == 'GAGNE' else "KO"
                            txt_ik = "-"
                            col_ik = "#666"
                            if 'User_Result' in r and pd.notna(r['User_Result']):
                                txt_ik = "OK" if r['User_Result'] == 'GAGNE' else "KO" if r['User_Result'] == 'PERDU' else "-"
                                col_ik = "#4ade80" if r['User_Result'] == 'GAGNE' else "#f87171" if r['User_Result'] == 'PERDU' else "#666"
                            html_table += f"<tr><td>{match_str}</td><td>{win_str}</td><td style='color:{col_ia}; font-weight:bold;'>{txt_ia}</td><td style='color:{col_ik}; font-weight:bold;'>{txt_ik}</td></tr>"
                        html_table += "</table>"
                        st.markdown(html_table, unsafe_allow_html=True)

# ==============================================================================
# TAB 2 : STATS (V9.3 COMPACT)
# ==============================================================================
with tab2:
    _, c_tab_center, _ = st.columns([1, 10, 1])
    
    with c_tab_center:
        # LECTURE CLOUD POUR STATS
        df = hist_df.copy()
        
        if not df.empty:
            finished_df = df[df['Result'].isin(['GAGNE', 'PERDU'])].copy()
            
            # 1. KPIs
            if not finished_df.empty:
                ia_win_count = len(finished_df[finished_df['Result'] == 'GAGNE'])
                ia_total = len(finished_df)
                ia_acc = (ia_win_count / ia_total) * 100
                
                user_played = finished_df[finished_df['User_Result'].isin(['GAGNE', 'PERDU'])]
                user_win_count = len(user_played[user_played['User_Result'] == 'GAGNE'])
                user_total = len(user_played)
                user_acc = (user_win_count / user_total * 100) if user_total > 0 else 0
                
                c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                with c_kpi1:
                    st.markdown(f"<div class='unified-card stat-box'><div class='stat-label'>IA SUCCESS</div><div class='stat-val color-ai'>{ia_acc:.1f}%</div><div class='stat-sub'>({ia_win_count}/{ia_total})</div></div>", unsafe_allow_html=True)
                with c_kpi2:
                    col_u = "color-ik" if user_acc >= ia_acc else "color-bad"
                    st.markdown(f"<div class='unified-card stat-box'><div class='stat-label'>IK SUCCESS</div><div class='stat-val {col_u}'>{user_acc:.1f}%</div><div class='stat-sub'>({user_win_count}/{user_total})</div></div>", unsafe_allow_html=True)
                
                # 2. CHART
                if 'User_Reason' in user_played.columns and not user_played.empty:
                    stats_reason = user_played.groupby('User_Reason')['User_Result'].apply(lambda x: (x == 'GAGNE').mean()).reset_index()
                    stats_reason.columns = ['Raison', 'Taux']
                    stats_reason['Count'] = user_played.groupby('User_Reason')['User_Result'].count().values
                    stats_reason = stats_reason[stats_reason['Count'] > 0]
                    
                    if not stats_reason.empty:
                        with st.expander("📊 Analyse des Raisons", expanded=True):
                            chart = alt.Chart(stats_reason).mark_bar().encode(
                                x=alt.X('Raison', axis=alt.Axis(title=None, labelAngle=0)),
                                y=alt.Y('Taux', axis=alt.Axis(format='%', title='Réussite')),
                                color=alt.condition(alt.datum.Taux >= 0.5, alt.value("#4ade80"), alt.value("#f87171")),
                                tooltip=['Raison', alt.Tooltip('Taux', format='.1%'), 'Count']
                            ).properties(height=200).configure_axis(grid=False)
                            st.altair_chart(chart, use_container_width=True)

            # 3. TABLEAU COMPACT (Style V8)
            st.markdown("---")
            
            df_disp = df.fillna("")
            df_disp['Date'] = pd.to_datetime(df_disp['Date'], errors='coerce')
            
            def merge_prono_res(prono, res):
                if not prono: return "-"
                code = get_short_code(get_clean_name(prono))
                if res not in ['GAGNE', 'PERDU']: return code
                icon = "✅" if res == 'GAGNE' else "❌"
                return f"{icon} {code}"

            # Noms courts pour compacité
            df_disp['Home'] = df_disp['Home'].apply(lambda x: get_short_code(get_clean_name(x)))
            df_disp['Away'] = df_disp['Away'].apply(lambda x: get_short_code(get_clean_name(x)))
            df_disp['Winner'] = df_disp['Real_Winner'].apply(lambda x: get_short_code(get_clean_name(x)) if x not in ["En attente...", ""] else "...")
            
            df_disp['Prono IA'] = df_disp.apply(lambda x: merge_prono_res(x['Predicted_Winner'], x['Result']), axis=1)
            
            if 'User_Prediction' not in df_disp.columns: df_disp['User_Prediction'] = ""
            if 'User_Result' not in df_disp.columns: df_disp['User_Result'] = ""
            if 'User_Reason' not in df_disp.columns: df_disp['User_Reason'] = ""
            
            df_disp['Prono IK'] = df_disp.apply(lambda x: merge_prono_res(x['User_Prediction'], x['User_Result']), axis=1)
            df_disp['Confidence'] = df_disp['Confidence']

            cols = ['Date', 'Home', 'Away', 'Winner', 'Prono IK', 'User_Reason', 'Prono IA', 'Confidence', 'Type']
            final_view = df_disp[cols].sort_index(ascending=False)
            final_view.insert(len(final_view.columns), "Del", False)
            
            # NOTE: En mode Cloud, l'édition du tableau est désactivée car elle n'écrit pas dans Supabase
            st.dataframe(final_view, hide_index=True, use_container_width=True)

# ==============================================================================
# TAB 3 : ADMIN
# ==============================================================================
with tab3:
    st.info("⚠️ Mode Cloud Native : Les données sont lues depuis Supabase.")
    st.caption("Pour mettre à jour, utilisez le script local 'GO_NBA.bat'.")
    
    st.markdown("---")
    st.subheader("🔮 Ajout Manuel (Local)")
    cm1, cm2, cm3 = st.columns(3)
    team_names = [f"{v['code']} - {v['full']}" for k,v in TEAMS_DB.items()]
    hm = cm1.selectbox("Home", team_names, index=None)
    aw = cm2.selectbox("Away", team_names, index=None)
    dt = cm3.date_input("Date", value=datetime.now())
    
    if hm and aw:
        if st.button("Analyser"):
            h_code = hm.split(' - ')[0]
            a_code = aw.split(' - ')[0]
            h_id = next(k for k,v in TEAMS_DB.items() if v['code'] == h_code)
            a_id = next(k for k,v in TEAMS_DB.items() if v['code'] == a_code)
            prob, _ = get_prediction(model, df_stats, h_id, a_id)
            if prob:
                win_name = TEAMS_DB[h_id]['full'] if prob > 0.5 else TEAMS_DB[a_id]['full']
                conf = prob*100 if prob > 0.5 else (1-prob)*100
                st.success(f"Vainqueur : {win_name} ({conf:.1f}%)")
                st.caption("Note : Pour sauvegarder, utilisez l'interface locale.")