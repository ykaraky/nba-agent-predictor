import pandas as pd
import os

HISTORY_FILE = 'data/bets_history.csv' # <-- Changement ici
GAMES_FILE = 'data/nba_games.csv'      # <-- Changement ici

print(f"\n--- VERIFICATION v5 ---")

if not os.path.exists(HISTORY_FILE):
    print("Pas d'historique.")
    exit()

bets = pd.read_csv(HISTORY_FILE)
if 'Real_Winner' not in bets.columns: bets['Real_Winner'] = None
bets['Date'] = pd.to_datetime(bets['Date'])

try:
    games = pd.read_csv(GAMES_FILE)
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games['MATCHUP'] = games['MATCHUP'].astype(str)
except:
    print("Pas de fichier NBA.")
    exit()

updated = 0
for index, bet in bets.iterrows():
    needs_update = (pd.isna(bet['Result']) or str(bet['Result']).strip() in ['None', ''] or str(bet['Real_Winner']) == "En attente...")
    
    if needs_update:
        pred_abbr = str(bet['Predicted_Winner']).split(' ')[0]
        games_date = games[games['GAME_DATE'] == bet['Date']]
        if games_date.empty: games_date = games[games['GAME_DATE'] == bet['Date'] + pd.Timedelta(days=1)]

        found_row = None
        for _, row in games_date.iterrows():
            if pred_abbr in row['MATCHUP'].replace('@', ' ').replace('vs.', ' '):
                row_str = str(row.values)
                if f"'{pred_abbr}'" in row_str or f" {pred_abbr} " in row_str:
                    found_row = row
                    break
        
        if found_row is not None:
            is_win = (found_row['WL'] == 'W')
            real_winner = str(bet['Predicted_Winner']) if is_win else (bet['Away'] if str(bet['Predicted_Winner']) == bet['Home'] else bet['Home'])
            bets.at[index, 'Result'] = "GAGNE" if is_win else "PERDU"
            bets.at[index, 'Real_Winner'] = real_winner
            updated += 1

if updated > 0:
    bets.to_csv(HISTORY_FILE, index=False)
    print(f"[OK] {updated} paris mis a jour.")
else:
    print("Rien a jour.")