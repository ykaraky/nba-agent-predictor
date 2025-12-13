import pandas as pd
import os

HISTORY_FILE = 'bets_history.csv'
GAMES_FILE = 'nba_games.csv'

print(f"\n--- VERIFICATION DES PARIS ---")

if not os.path.exists(HISTORY_FILE):
    print("Pas d'historique de paris trouve pour l'instant.")
    exit()

# 1. Charger l'historique
bets = pd.read_csv(HISTORY_FILE)
bets['Date'] = pd.to_datetime(bets['Date'])

# 2. Charger les résultats réels
games = pd.read_csv(GAMES_FILE)
games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
games = games[games['GAME_DATE'] >= '2023-01-01']

updated_count = 0

# 3. Boucle de vérification
for index, bet in bets.iterrows():
    if pd.isna(bet['Result']) or bet['Result'] == '':
        
        match_reel = games[
            (games['GAME_DATE'] == bet['Date']) & 
            (games['MATCHUP'].str.contains(bet['Home'])) & 
            (games['MATCHUP'].str.contains('vs.'))
        ]
        
        if len(match_reel) > 0:
            real_result = match_reel.iloc[0]
            real_winner = bet['Home'] if real_result['WL'] == 'W' else bet['Away']
            status = "GAGNE" if bet['Predicted_Winner'] == real_winner else "PERDU"
            
            bets.at[index, 'Result'] = status
            # Suppression émojis
            print(f"[RES] Match du {bet['Date'].date()} ({bet['Home']} vs {bet['Away']}) : {status}")
            updated_count += 1

# 4. Sauvegarde
if updated_count > 0:
    bets.to_csv(HISTORY_FILE, index=False)
    print(f"\n[OK] {updated_count} paris mis a jour dans l'historique.")
else:
    print("Aucun nouveau resultat trouve.")

# 5. Stats
completed_bets = bets.dropna(subset=['Result'])
total_completed = len(completed_bets)

if total_completed > 0:
    nb_gagnes = len(completed_bets[completed_bets['Result'] == 'GAGNE'])
    accuracy = (nb_gagnes / total_completed) * 100
    
    print("\n--- BILAN DE L'AGENT ---")
    print(f"Total Paris : {total_completed}")
    print(f"Victoires   : {nb_gagnes}")
    print(f"Defaites    : {total_completed - nb_gagnes}")
    print(f"PRECISION   : {accuracy:.1f}%")