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
    /* Carte de match style Dashboard */
    .match-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #3e3e3e;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .team-label { color: #ccc; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px;}
    
    /* Nouveau style pour le score */
    .score-box {
        font-size: 1.8em;
        font-weight: 800;
        padding: 5px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    .win-green { color: #4ade80; border-bottom: 2px solid #4ade80; }
    .win-red { color: #f87171; border-bottom: 2px solid #f87171; }
    
    .section-title { font-size: 1.2em; font-weight: bold; margin-top: 20px; margin-bottom: 15px; color: #fff; border-left: 4px solid #00d4ff; padding-left: 10px; }
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

def show_logo(team_id, width=55):
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
        with open(HISTORY_FILE, 'w') as f: f.write("Date,Home,Away,Predicted_Winner,Confidence,Type,Result,Real_Winner\n")
    try:
        df = pd.read_csv(HISTORY_FILE)
        if not df[(df['Date'] == date) & (df['Home'] == h_name) & (df['Away'] == a_name)].empty: return
    except: pass
    with open(HISTORY_FILE, 'a') as f:
        f.write(f"\n{date},{h_name},{a_name},{w_name},{conf:.1f}%,Auto,,")

def get_last_mod(filepath):
    if os.path.exists(filepath):
        return datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%d/%m %H:%M')
    return "N/A"

# Fonction Scan
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
        
        # B. HISTORIQUE
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
# TAB 1 : MATCHS (CLEAN DESIGN)
# ==============================================================================
with tab1:
    
    # Auto Scan silencieux
    if not st.session_state['schedule_data']:
        st.session_state['schedule_data'] = scan_schedule()
        st.session_state['last_update'] = datetime.now().strftime('%H:%M')

    if st.session_state['last_update']:
        st.caption(f"Dernier scan : {st.session_state['last_update']}")

    # Affichage
    schedule = st.session_state.get('schedule_data', {})
    
    if schedule:
        for date_key, dfs_list in schedule.items():
            is_today = date_key == datetime.now().strftime('%Y-%m-%d')
            titre = f"Affiches de ce soir ({date_key})" if is_today else f"Affiches du {date_key}"
            
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
                    
                    # API
                    if 'HOME_TEAM_ID' in row:
                        h_id, a_id = row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']
                        if h_id in TEAMS_DB: h_name = TEAMS_DB[h_id]['full']
                        if a_id in TEAMS_DB: a_name = TEAMS_DB[a_id]['full']
                        
                        match_id = f"{h_name} vs {a_name}"
                        if match_id in seen_matches: continue
                        seen_matches.append(match_id)
                        
                        prob, det = get_prediction(model, df_stats, h_id, a_id)
                        if prob:
                            w = h_name if prob > 0.5 else a_name
                            c = prob*100 if prob > 0.5 else (1-prob)*100
                            save_bet_auto(date_key, h_name, a_name, w, c)

                    # MANUEL
                    elif 'Home' in row:
                        source_type = "History"
                        h_name, a_name = row['Home'], row['Away']
                        match_id = f"{h_name} vs {a_name}"
                        if match_id in seen_matches: continue
                        seen_matches.append(match_id)
                        h_id = next((k for k, v in TEAMS_DB.items() if v['full'] == get_clean_name(h_name)), 0)
                        a_id = next((k for k, v in TEAMS_DB.items() if v['full'] == get_clean_name(a_name)), 0)
                        winner = row['Predicted_Winner']
                        try:
                            conf_val = float(str(row['Confidence']).replace('%', ''))/100
                            is_home_winner = (winner == h_name)
                            prob = conf_val if is_home_winner else (1-conf_val)
                            det = {'rh':0, 'ra':0}
                        except: prob = None

                    # RENDU CARTE
                    if prob is not None and h_id != 0:
                        with cols[card_count % 2]:
                            with st.container(border=True):
                                is_h_win = prob > 0.5
                                val_disp = prob*100 if is_h_win else (1-prob)*100
                                col_txt = "win-green" if is_h_win else "win-red"
                                
                                c1, c2, c3 = st.columns([1,2,1])
                                with c1:
                                    show_logo(h_id)
                                    st.caption(TEAMS_DB.get(h_id, {}).get('nick', h_name))
                                with c3:
                                    show_logo(a_id)
                                    st.caption(TEAMS_DB.get(a_id, {}).get('nick', a_name))
                                with c2:
                                    # DESIGN CHEVRON
                                    arr = "❮" if is_h_win else ""
                                    arr_r = "❯" if not is_h_win else ""
                                    
                                    st.markdown(f"""
                                    <div style='text-align: center; margin-top: 10px;'>
                                        <span class='score-box {col_txt}'>{arr} {val_disp:.0f}% {arr_r}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if source_type == "History": st.caption("Manuel")
                        card_count += 1

    elif st.session_state['schedule_data'] == {}:
        st.warning("Aucun match trouvé pour les 7 prochains jours.")

    st.write("")
    
    # BILAN FLASH
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        finished = hist[hist['Result'].isin(['GAGNE', 'PERDU'])].copy()
        if not finished.empty:
            st.markdown("<div class='section-title'>Derniers Résultats</div>", unsafe_allow_html=True)
            dates = sorted(finished['Date'].unique(), reverse=True)[:2]
            for d in dates:
                day_rows = finished[finished['Date'] == d]
                wins = len(day_rows[day_rows['Result'] == 'GAGNE'])
                with st.expander(f"📅 {d} ({wins}/{len(day_rows)})", expanded=True):
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
        
        df_hist['Home_Clean'] = df_hist['Home'].apply(get_clean_name)
        df_hist['Away_Clean'] = df_hist['Away'].apply(get_clean_name)
        df_hist['Prono_Short'] = df_hist['Predicted_Winner'].apply(lambda x: get_short_code(get_clean_name(x)))
        df_hist['Winner_Short'] = df_hist['Real_Winner'].apply(lambda x: get_short_code(get_clean_name(x)) if x not in ["En attente...", "None", ""] else "...")
        
        display_df = df_hist[['Date', 'Home_Clean', 'Away_Clean', 'Prono_Short', 'Winner_Short', 'Result', 'Confidence', 'Type']].copy()
        display_df.columns = ['Date', 'Home', 'Away', 'Prono', 'Vainqueur', 'Result', 'Confidence', 'Type']
        display_df = display_df.sort_index(ascending=False)
        display_df.insert(len(display_df.columns), "Del", False)
        
        edited = st.data_editor(
            display_df,
            column_config={
                "Del": st.column_config.CheckboxColumn("🗑️", width="small"),
                "Result": st.column_config.TextColumn("Res"),
                "Confidence": st.column_config.TextColumn("Conf"),
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
                cols_save = ['Date', 'Home', 'Away', 'Predicted_Winner', 'Confidence', 'Type', 'Result', 'Real_Winner']
                hist_new[cols_save].to_csv(HISTORY_FILE, index=False)
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
                st.session_state['games_data'] = None
                st.session_state['schedule_data'] = {} 
                load_resources.clear()
                s.update(label="Terminé", state="complete")
                st.rerun()
        if st.button("Mise à jour Hebdo (Lundi)"):
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