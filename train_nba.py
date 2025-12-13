import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def train_model():
    """
    Fonction principale d'entraînement appelée par l'interface.
    Retourne (Succès, Message, Précision)
    """
    print("--- Démarrage Entraînement ---")
    
    try:
        # Vérification des fichiers
        if not os.path.exists('nba_games_ready.csv'):
            return False, "Fichier 'nba_games_ready.csv' introuvable.", 0

        df = pd.read_csv('nba_games_ready.csv')

        # Séparation Domicile / Extérieur
        df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
        df_home = df[df['IS_HOME'] == True].copy().add_suffix('_HOME')
        df_away = df[df['IS_HOME'] == False].copy().add_suffix('_AWAY')
        
        # Correction des noms de colonnes pour la fusion
        df_home = df_home.rename(columns={'GAME_ID_HOME': 'GAME_ID'})
        df_away = df_away.rename(columns={'GAME_ID_AWAY': 'GAME_ID'})
        
        df_final = pd.merge(df_home, df_away, on='GAME_ID')

        # Création des Features (Les ingrédients)
        df_final['DIFF_EFG'] = df_final['EFG_PCT_LAST_5_HOME'] - df_final['EFG_PCT_LAST_5_AWAY']
        df_final['DIFF_TOV'] = df_final['TOV_PCT_LAST_5_HOME'] - df_final['TOV_PCT_LAST_5_AWAY']
        df_final['DIFF_ORB'] = df_final['ORB_RAW_LAST_5_HOME'] - df_final['ORB_RAW_LAST_5_AWAY']
        df_final['DIFF_WIN'] = df_final['WIN_LAST_5_HOME'] - df_final['WIN_LAST_5_AWAY']
        df_final['DIFF_REST'] = df_final['DAYS_REST_HOME'] - df_final['DAYS_REST_AWAY']

        features = [
            'EFG_PCT_LAST_5_HOME', 'EFG_PCT_LAST_5_AWAY',
            'TOV_PCT_LAST_5_HOME', 'TOV_PCT_LAST_5_AWAY',
            'ORB_RAW_LAST_5_HOME', 'ORB_RAW_LAST_5_AWAY',
            'DIFF_EFG', 'DIFF_TOV', 'DIFF_ORB', 'DIFF_WIN', 'DIFF_REST'
        ]
        target = 'WIN_HOME'

        X = df_final[features]
        y = df_final[target]

        # Entraînement
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Évaluation
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Sauvegarde
        model.save_model("nba_predictor.json")
        
        return True, "Modèle entraîné et sauvegardé avec succès.", acc

    except Exception as e:
        return False, f"Erreur technique : {str(e)}", 0