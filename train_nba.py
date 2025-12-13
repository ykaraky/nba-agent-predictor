import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("--- Entraînement IA : Modèle FOUR FACTORS ---")

df = pd.read_csv('nba_games_ready.csv')

# Séparation Domicile / Extérieur
df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
df_home = df[df['IS_HOME'] == True].copy().add_suffix('_HOME')
df_away = df[df['IS_HOME'] == False].copy().add_suffix('_AWAY')
df_home = df_home.rename(columns={'GAME_ID_HOME': 'GAME_ID'})
df_away = df_away.rename(columns={'GAME_ID_AWAY': 'GAME_ID'})
df_final = pd.merge(df_home, df_away, on='GAME_ID')

# --- NOUVELLES FEATURES ---
# On compare les Four Factors de l'équipe Home vs l'équipe Away
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

# Résultats
acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n✅ Nouvelle Précision (Four Factors) : {acc:.2%}")

# Importance des facteurs
imp = pd.DataFrame({'Facteur': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
print("\nTop Facteurs Décisifs :")
print(imp.head(5))

model.save_model("nba_predictor.json")