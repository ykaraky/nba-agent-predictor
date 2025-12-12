import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("--- Entraînement avec gestion de la FATIGUE ---")

# 1. Chargement
df = pd.read_csv('nba_games_ready.csv')

# 2. Séparation Domicile / Extérieur
df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
df_home = df[df['IS_HOME'] == True].copy().add_suffix('_HOME')
df_away = df[df['IS_HOME'] == False].copy().add_suffix('_AWAY')

df_home = df_home.rename(columns={'GAME_ID_HOME': 'GAME_ID'})
df_away = df_away.rename(columns={'GAME_ID_AWAY': 'GAME_ID'})

df_final = pd.merge(df_home, df_away, on='GAME_ID')

# 3. Création des Features (Ce que l'IA regarde)
# On ajoute la différence de repos (ex: Home a 2 jours, Away a 0 jour = Avantage Home)
df_final['DIFF_PTS'] = df_final['PTS_LAST_5_HOME'] - df_final['PTS_LAST_5_AWAY']
df_final['DIFF_REST'] = df_final['DAYS_REST_HOME'] - df_final['DAYS_REST_AWAY']

# LISTE MISE À JOUR AVEC LA FATIGUE
features = [
    'PTS_LAST_5_HOME', 'PTS_LAST_5_AWAY',
    'WIN_LAST_5_HOME', 'WIN_LAST_5_AWAY',
    'DAYS_REST_HOME', 'DAYS_REST_AWAY',  # <--- NOUVEAU
    'DIFF_PTS', 'DIFF_REST'              # <--- NOUVEAU
]
target = 'WIN_HOME'

X = df_final[features]
y = df_final[target]

# 4. Entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 5. Résultats
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"\nPrécision du modèle (Fatigue incluse) : {acc:.2%}")

# Vérification de l'importance de la fatigue
imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
print(imp.sort_values(by='Importance', ascending=False))

model.save_model("nba_predictor.json")