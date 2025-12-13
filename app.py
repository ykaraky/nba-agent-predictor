import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime
import os
import sys
import subprocess
import time
from nba_api.stats.static import teams

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="NBA AI Predictor",
    page_icon="🏀",
    layout="wide"
)

# --- FONCTIONS UTILES (CACHÉES) ---

# Le cache permet de ne pas tout recharger à chaque clic
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    try:
        model.load_model("nba_predictor.json")
        return model
    except:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('nba_games_ready.csv')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df
    except:
        return None

def get_team_list():
    nba_teams = teams.get_teams()
    # On crée une liste "LAL - Los Angeles Lakers" pour le menu déroulant
    team_dict = {f"{t['abbreviation']} - {t['nickname']}": t['id'] for t in nba_teams}
    return team_dict

def run_update_script(script_name):
    """Lance un script externe (data, features, etc.)"""
    try:
        result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

# --- LOGIQUE DE PRÉDICTION (Four Factors) ---
def get_prediction(model, df_history, home_id, away_id):
    # Filtrer l'historique pour les équipes
    home_games = df_history[df_history['TEAM_ID'] == home_id].sort_values('GAME_DATE')
    away_games = df_history[df_history['TEAM_ID'] == away_id].sort_values('GAME_DATE')
    
    if home_games.empty or away_games.empty:
        return None, "Données manquantes"

    # Dernier match connu
    last_home = home_games.iloc[-1]
    last_away = away_games.iloc[-1]
    
    # Calcul date et repos
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    rest_home = (today - last_home['GAME_DATE']).days
    rest_away = (today - last_away['GAME_DATE']).days
    rest_home = min(rest_home, 7)
    rest_away = min(rest_away, 7)

    # Création du DataFrame pour l'IA (Mêmes colonnes que l'entraînement !)
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
        'DIFF_REST': rest_home - rest_away
    }])

    # Prédiction
    probs = model.predict_proba(input_data)[0]
    prob_home = probs[1]
    
    return prob_home, {
        'home_stats': last_home,
        'away_stats': last_away,
        'rest_home': rest_home,
        'rest_away': rest_away
    }

# --- INTERFACE GRAPHIQUE ---

st.title("🏀 NBA Agent : Le Cerveau")
st.markdown("Bienvenue dans ton centre de commande. Analyse, Prédiction et Tracking.")

# Chargement initial
model = load_model()
df = load_data()
teams_dict = get_team_list()

if model is None or df is None:
    st.error("⚠️ Fichiers manquants (json ou csv). Lance le launcher une fois d'abord !")
    st.stop()

# Onglets
tab1, tab2, tab3 = st.tabs(["🔮 Pronostics", "📊 Bilan & Tracking", "🔄 Mise à jour"])

# --- ONGLET 1 : PRONOSTICS ---
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Domicile")
        # Menu déroulant intelligent
        home_choice = st.selectbox("Choisir équipe Domicile", list(teams_dict.keys()), index=13) # index 13 = LAL par défaut souvent
        
    with col2:
        st.subheader("Extérieur")
        away_choice = st.selectbox("Choisir équipe Extérieur", list(teams_dict.keys()), index=1) # index 1 = BOS

    # Bouton d'action
    if st.button("Lancer l'analyse 🚀", type="primary"):
        home_id = teams_dict[home_choice]
        away_id = teams_dict[away_choice]
        
        prob_home, details = get_prediction(model, df, home_id, away_id)
        
        if prob_home is not None:
            # Affichage visuel du résultat
            st.divider()
            
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c1:
                st.write(f"**{home_choice.split('-')[0]}**")
                st.caption(f"Repos: {details['rest_home']}j")
                # Affichage des stats clés
                st.metric("Adresse (eFG%)", f"{details['home_stats']['EFG_PCT_LAST_5']:.1%}")
                st.metric("Pertes balle", f"{details['home_stats']['TOV_PCT_LAST_5']:.1%}")
                
            with c2:
                # Le gros résultat au milieu
                if prob_home > 0.5:
                    winner = home_choice.split('-')[0]
                    conf = prob_home * 100
                    color = "green"
                else:
                    winner = away_choice.split('-')[0]
                    conf = (1 - prob_home) * 100
                    color = "red"
                
                st.markdown(f"<h1 style='text-align: center; color: {color};'>{winner}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>Confiance : {conf:.1f}%</h3>", unsafe_allow_html=True)
                
                # Barre de progression visuelle
                st.progress(int(conf), text="Indice de confiance")
                
                if details['rest_home'] <= 1: st.warning(f"⚠️ {home_choice.split('-')[0]} est en Back-to-back !")
                if details['rest_away'] <= 1: st.warning(f"⚠️ {away_choice.split('-')[0]} est en Back-to-back !")

            with c3:
                st.write(f"**{away_choice.split('-')[0]}**")
                st.caption(f"Repos: {details['rest_away']}j")
                st.metric("Adresse (eFG%)", f"{details['away_stats']['EFG_PCT_LAST_5']:.1%}")
                st.metric("Pertes balle", f"{details['away_stats']['TOV_PCT_LAST_5']:.1%}")
                
        else:
            st.error("Erreur de calcul.")

# --- ONGLET 2 : BILAN ---
with tab2:
    st.header("Historique des paris")
    
    if os.path.exists('bets_history.csv'):
        history = pd.read_csv('bets_history.csv')
        
        # Filtres
        st.dataframe(history.tail(10)) # Montre les 10 derniers

        # --- AJOUT : BOUTON NETTOYAGE ---
        st.divider() # Une ligne de séparation jolie
        
        col_clean, col_void = st.columns([1, 3]) # On fait une petite colonne pour le bouton
        with col_clean:
            if st.button("🧹 Supprimer les doublons"):
                # 1. On charge tout le fichier
                df_clean = pd.read_csv('bets_history.csv')
                
                # 2. On supprime les lignes identiques
                # On regarde si Date + Home + Away sont identiques
                # keep='last' garde la dernière version (la plus récente)
                df_clean = df_clean.drop_duplicates(subset=['Date', 'Home', 'Away'], keep='last')
                
                # 3. On sauvegarde
                df_clean.to_csv('bets_history.csv', index=False)
                
                # 4. Message et Rechargement de la page
                st.toast("Nettoyage effectué !", icon="✨")
                time.sleep(1) # Petite pause pour voir le message
                st.rerun()    # Commande magique pour rafraîchir l'interface
        
        # Calcul des stats en direct
        if 'Result' in history.columns:
            finished = history.dropna(subset=['Result'])
            finished = finished[finished['Result'] != '']
            
            if not finished.empty:
                wins = len(finished[finished['Result'] == 'GAGNÉ'])
                total = len(finished)
                acc = (wins / total) * 100
                
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Paris terminés", total)
                kpi2.metric("Victoires", wins)
                kpi3.metric("Précision (Accuracy)", f"{acc:.1f}%")
                
                if acc > 55:
                    st.success("🔥 L'IA est rentable !")
                elif acc > 50:
                    st.warning("⚖️ L'IA est à l'équilibre.")
                else:
                    st.error("❄️ L'IA perd de l'argent.")
            else:
                st.info("Pas encore de résultats validés.")
    else:
        st.write("Pas encore d'historique.")

# --- ONGLET 3 : ACTIONS ---
with tab3:
    st.header("Centre de maintenance")
    st.write("Utilise ces boutons pour mettre à jour les données manuellement.")
    
    col_up, col_feat, col_verif = st.columns(3)
    
    with col_up:
        if st.button("1. Télécharger Matchs 📥"):
            with st.spinner("Téléchargement en cours..."):
                log = run_update_script('data_nba.py')
                st.success("Terminé !")
                with st.expander("Voir les logs"):
                    st.code(log)
                    
    with col_feat:
        if st.button("2. Calculer Stats 🧮"):
            with st.spinner("Calcul des Four Factors..."):
                log = run_update_script('features_nba.py')
                st.success("Terminé !")
                # On vide le cache pour recharger les nouvelles données tout de suite
                load_data.clear()
                
    with col_verif:
        if st.button("3. Vérifier Paris ✅"):
            with st.spinner("Vérification des résultats..."):
                log = run_update_script('verify_bets.py')
                st.success("Terminé !")
                with st.expander("Voir les logs"):
                    st.code(log)