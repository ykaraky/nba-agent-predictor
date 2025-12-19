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

# CSS HACKS pour UX/UI demandée
st.markdown("""
<style>
    /* 1. Header Fixed & Tabs Styling */
    .stAppHeader {
        background-color: #0e1117;
        opacity: 0.95;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: #0e1117;
        padding-top: 10px;
        padding-bottom: 10px;
        position: sticky;
        top: 3rem; /* Ajuster selon header */
        z-index: 999;
        border-bottom: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px;
        font-size: 1.1em;
        font-weight: 600;
        color: #888;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #00d4ff;
        background-color: rgba(0, 212, 255, 0.1);
    }
    
    /* 2. Cards Compactes */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    .match-card-container {
        background-color: #1c1f26;
        border-radius: 10px;
        padding: 15px 10px;
        margin-bottom: 10px;
        border: 1px solid #2e3035;
    }
    
    /* 3. Textes & Infos */
    .team-name { font-weight: bold; font-size: 1.1em; margin-bottom: 0px; }
    .team-rank { font-size: 0.8em; color: #aaa; }
    .score-badge { 
        background-color: #333; 
        color: #fff; 
        padding: 2px 6px; 
        border-radius: 4px; 
        font-size: 0.75em; 
        font-weight: bold;
    }
    .ia-badge {
        font-size: 0.7em; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
        color: #666;
        margin-bottom: 2px;
    }
    .ia-choice {
        font-size: 1.2em; 
        font-weight: 900; 
        color: #00d4ff;
    }
    
    /* Cacher les icônes de lien Streamlit */
    .css-15zrgzn {display: none;}
    
</style>
""", unsafe_allow_html=True)

# --- 2. SESSIONS & CHEMINS ---
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
        if df[mask].empty:
            st.error("Match introuvable.")
            return
        idx = df[mask].index[0]
        df.at[idx, 'User_Prediction'] = user_choice
        df.to_csv(HISTORY_FILE, index=False)
        # Pas de toast intrusif, juste un rerun fluide
    except Exception as e:
        st.error(f"Erreur : {e}")

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
id_to_name = {k: v['full'] for k, v in TEAMS_DB.items()}

# --- HEADER (Fixe via CSS, Contenu ici) ---
c_head1, c_head2 = st.columns([1, 8])
with c_head1:
    if os.path.exists(APP_LOGO): st.image(APP_LOGO, width=80)
    else: st.title("🏀")
with c_head2:
    st.markdown("<h2 style='margin-top:0px; margin-bottom:0px;'>NBA AGENT PREDIKTOR</h2>", unsafe_allow_html=True)
    st.caption("Intelligence Artificielle vs Intuition Humaine")

# --- NAVIGATION ---
# Texte simple, sans icônes pour le style "Clean"
tab1, tab2, tab3 = st.tabs(["MATCHS", "STATS", "ADMIN"])

# ==============================================================================
# TAB 1 : MATCHS
# ==============================================================================
with tab1:
    if not st.session_state['schedule_data']:
        with st.spinner("Analyse du calendrier..."):
            st.session_state['schedule_data'] = scan_schedule()
            st.session_state['last_update'] = datetime.now().strftime('%H:%M')

    schedule = st.session_state.get('schedule_data', {})
    
    if schedule:
        hist_df = pd.DataFrame()
        if os.path.exists(HISTORY_FILE): hist_df = pd.read_csv(HISTORY_FILE)

        for date_key, dfs_list in schedule.items():
            is_today = date_key == datetime.now().strftime('%Y-%m-%d')
            try:
                date_obj = datetime.strptime(date_key, '%Y-%m-%d')
                date_fmt = date_obj.strftime('%d.%m.%Y')
            except: date_fmt = date_key
            
            # Titre Section
            st.markdown(f"#### {'🔥 Ce Soir' if is_today else '📅 ' + date_fmt}")
            
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

                    # RENDU CARTE (Design V6.1)
                    if prob is not None and h_id != 0:
                        # Calculs IA
                        is_h_win = prob > 0.5
                        ia_conf = prob*100 if is_h_win else (1-prob)*100
                        ia_short = TEAMS_DB.get(h_id if is_h_win else a_id, {}).get('code', 'IA')
                        
                        # Infos Teams
                        inf_h = STANDINGS_DB.get(h_id, {'rec': '', 'strk': '', 'rank': ''})
                        inf_a = STANDINGS_DB.get(a_id, {'rec': '', 'strk': '', 'rank': ''})
                        
                        # Couleurs Série
                        c_strk_h = "#4ade80" if 'W' in inf_h['strk'] else "#f87171"
                        c_strk_a = "#4ade80" if 'W' in inf_a['strk'] else "#f87171"

                        with st.container():
                            # Utilisation container pour fond CSS (classe match-card-container a ajouter si on veut pousser le CSS)
                            # Layout : [ TEAM A (35%) | CENTER (30%) | TEAM B (35%) ]
                            c1, c2, c3 = st.columns([3.5, 3, 3.5])
                            
                            # --- TEAM HOME (Gauche) ---
                            with c1:
                                rc1, rc2 = st.columns([1, 2])
                                with rc1: show_logo(h_id, width=50)
                                with rc2:
                                    st.markdown(f"<div class='team-name'>{TEAMS_DB.get(h_id,{}).get('full', h_name)}</div>", unsafe_allow_html=True)
                                    if inf_h['rec']:
                                        st.markdown(f"<span class='team-rank'>#{inf_h['rank']} ({inf_h['rec']})</span> <span class='score-badge' style='color:{c_strk_h}'>{inf_h['strk']}</span>", unsafe_allow_html=True)

                            # --- CENTER (Duel Zone) ---
                            with c2:
                                # Partie IA
                                st.markdown(f"<div style='text-align:center;'>", unsafe_allow_html=True)
                                st.markdown(f"<div class='ia-badge'>🤖 PRONO IA</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='ia-choice'>{ia_short} <span style='font-size:0.6em; color:#888;'>{ia_conf:.0f}%</span></div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Partie USER (Boutons Compacts)
                                bc1, bc2 = st.columns(2)
                                code_h = TEAMS_DB.get(h_id, {}).get('code', 'HOM')
                                code_a = TEAMS_DB.get(a_id, {}).get('code', 'AWY')
                                
                                # Bouton Home
                                # Style: Si selectionné = Primary (couleur theme), sinon Secondary (gris)
                                t_h = "primary" if user_bet_val == h_name else "secondary"
                                if bc1.button(code_h, key=f"bH_{match_id}", type=t_h, use_container_width=True):
                                    save_user_vote(date_key, h_name, a_name, h_name)
                                    st.rerun()
                                
                                # Bouton Away
                                t_a = "primary" if user_bet_val == a_name else "secondary"
                                if bc2.button(code_a, key=f"bA_{match_id}", type=t_a, use_container_width=True):
                                    save_user_vote(date_key, h_name, a_name, a_name)
                                    st.rerun()

                            # --- TEAM AWAY (Droite) ---
                            with c3:
                                rc1, rc2 = st.columns([2, 1])
                                with rc1:
                                    st.markdown(f"<div class='team-name' style='text-align:right;'>{TEAMS_DB.get(a_id,{}).get('full', a_name)}</div>", unsafe_allow_html=True)
                                    if inf_a['rec']:
                                        st.markdown(f"<div style='text-align:right;'><span class='score-badge' style='color:{c_strk_a}'>{inf_a['strk']}</span> <span class='team-rank'>({inf_a['rec']}) #{inf_a['rank']}</span></div>", unsafe_allow_html=True)
                                with rc2: show_logo(a_id, width=50)

                            st.divider()

    elif st.session_state['schedule_data'] == {}:
        st.warning("Aucun match trouvé.")

    # 4. BILAN DE LA NUIT (NOUVEAU)
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        # On regarde les matchs "finis"
        finished = hist[hist['Result'].isin(['GAGNE', 'PERDU'])].copy()
        
        if not finished.empty:
            st.markdown("#### 🏁 Derniers Résultats")
            dates = sorted(finished['Date'].unique(), reverse=True)[:2]
            
            for d in dates:
                day_rows = finished[finished['Date'] == d]
                
                # Calcul Scores IA vs IK
                ia_wins = len(day_rows[day_rows['Result'] == 'GAGNE'])
                # User wins : il faut que User_Result soit GAGNE
                user_wins = 0
                if 'User_Result' in day_rows.columns:
                    user_wins = len(day_rows[day_rows['User_Result'] == 'GAGNE'])
                
                total = len(day_rows)
                
                try: d_fmt = datetime.strptime(d, '%Y-%m-%d').strftime('%d.%m.%Y')
                except: d_fmt = d
                
                with st.expander(f"{d_fmt} | 🤖 IA: {ia_wins}/{total} | 👤 IK: {user_wins}/{total}", expanded=True):
                    # Petit tableau récapitulatif clean
                    for _, r in day_rows.iterrows():
                        # Icones IA
                        ia_icon = "✅" if r['Result'] == "GAGNE" else "❌"
                        # Icones IK
                        ik_icon = "➖" # Par défaut si pas joué
                        if 'User_Result' in r and pd.notna(r['User_Result']):
                            if r['User_Result'] == 'GAGNE': ik_icon = "✅"
                            elif r['User_Result'] == 'PERDU': ik_icon = "❌"
                        
                        match_str = f"{get_short_code(r['Home'])} vs {get_short_code(r['Away'])}"
                        win_str = get_short_code(r['Real_Winner']) if pd.notna(r['Real_Winner']) else "?"
                        
                        # Ligne : IA | IK | Match | Vainqueur
                        c1, c2, c3, c4 = st.columns([1, 1, 3, 1])
                        c1.write(f"🤖 {ia_icon}")
                        c2.write(f"👤 {ik_icon}")
                        c3.caption(match_str)
                        c4.caption(f"Win: {win_str}")

# ==============================================================================
# TAB 2 : STATS (COMPACT)
# ==============================================================================
with tab2:
    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        df_hist = df_hist.fillna("")
        df_hist['Date'] = pd.to_datetime(df_hist['Date'], errors='coerce')
        
        # PREPARATION DONNEES COMPACTES
        # 1. Nettoyage Noms
        df_hist['H'] = df_hist['Home'].apply(lambda x: get_short_code(get_clean_name(x)))
        df_hist['A'] = df_hist['Away'].apply(lambda x: get_short_code(get_clean_name(x)))
        df_hist['Win'] = df_hist['Real_Winner'].apply(lambda x: get_short_code(get_clean_name(x)) if x not in ["En attente...", ""] else "...")
        
        # 2. IA Data
        df_hist['IA_P'] = df_hist['Predicted_Winner'].apply(lambda x: get_short_code(get_clean_name(x)))
        # Transformation Resultat IA en Bool ou Icone
        df_hist['IA_R'] = df_hist['Result'].apply(lambda x: True if x == 'GAGNE' else False if x == 'PERDU' else None)
        
        # 3. User Data
        if 'User_Prediction' not in df_hist.columns: df_hist['User_Prediction'] = ""
        df_hist['IK_P'] = df_hist['User_Prediction'].apply(lambda x: get_short_code(get_clean_name(x)) if x != "" else "-")
        
        if 'User_Result' not in df_hist.columns: df_hist['User_Result'] = ""
        df_hist['IK_R'] = df_hist['User_Result'].apply(lambda x: True if x == 'GAGNE' else False if x == 'PERDU' else None)

        # 4. Selection Colonnes
        display_df = df_hist[['Date', 'H', 'A', 'Win', 'IK_P', 'IK_R', 'IA_P', 'IA_R', 'Confidence', 'Type']].copy()
        
        # Tri
        display_df = display_df.sort_index(ascending=False)
        display_df.insert(len(display_df.columns), "Del", False)
        
        # TABLEAU
        edited = st.data_editor(
            display_df,
            column_config={
                "Del": st.column_config.CheckboxColumn("🗑️", width="small"),
                "Date": st.column_config.DateColumn("Date", format="DD.MM"),
                "H": st.column_config.TextColumn("Dom", width="small"),
                "A": st.column_config.TextColumn("Ext", width="small"),
                "Win": st.column_config.TextColumn("Vainq", width="small"),
                
                "IK_P": st.column_config.TextColumn("Moi", width="small", help="Mon Prono"),
                "IK_R": st.column_config.CheckboxColumn("R", width="small"), # Checkbox read-only visual hack
                
                "IA_P": st.column_config.TextColumn("IA", width="small", help="Prono IA"),
                "IA_R": st.column_config.CheckboxColumn("R", width="small"),
                
                "Confidence": st.column_config.TextColumn("%", width="small"),
                "Type": st.column_config.TextColumn("Typ", width="small"),
            },
            height=600, # Hauteur fixe pour eviter double scroll
            hide_index=True,
            use_container_width=True
        )
        
        # Actions
        if st.button("Supprimer la sélection", type="primary"):
            to_del_idx = edited[edited.Del == True].index
            if not to_del_idx.empty:
                hist_new = df_hist.drop(to_del_idx)
                # Reconstruction CSV propre
                cols_save = ['Date', 'Home', 'Away', 'Predicted_Winner', 'Confidence', 'Type', 'Result', 'Real_Winner', 'User_Prediction', 'User_Result']
                hist_new['Date'] = hist_new['Date'].dt.strftime('%Y-%m-%d')
                # Mapping inverse si besoin ou rechargement depuis original (plus sur)
                # Astuce: On recharge l'original et on drop par index
                orig = pd.read_csv(HISTORY_FILE)
                orig.drop(to_del_idx, inplace=True)
                orig.to_csv(HISTORY_FILE, index=False)
                st.success("Supprimé !"); time.sleep(0.5); st.rerun()

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