import pandas as pd
import os

HISTORY_FILE = 'bets_history.csv'
GAMES_FILE = 'nba_games.csv'

print(f"\n--- VERIFICATION INTELLIGENTE v3.1 ---")

if not os.path.exists(HISTORY_FILE):
    print("Pas d'historique.")
    exit()

bets = pd.read_csv(HISTORY_FILE)

# Création de la colonne si elle n'existe pas dans le CSV
if 'Real_Winner' not in bets.columns:
    bets['Real_Winner'] = None

bets['Date'] = pd.to_datetime(bets['Date'])

try:
    games = pd.read_csv(GAMES_FILE)
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games['MATCHUP'] = games['MATCHUP'].astype(str)
except:
    print("Pas de fichier NBA.")
    exit()

updated_count = 0

for index, bet in bets.iterrows():
    # CONDITION MISE À JOUR : On vérifie si le Résultat manque OU si le Vainqueur Réel manque
    # Cela force la mise à jour des anciennes lignes incomplètes
    needs_update = (
        pd.isna(bet['Result']) or 
        str(bet['Result']).strip() in ['None', ''] or 
        pd.isna(bet['Real_Winner']) or 
        str(bet['Real_Winner']) == "En attente..."
    )

    if needs_update:
        pred_full = str(bet['Predicted_Winner'])
        pred_abbr = pred_full.split(' ')[0] 
        
        # Recherche du match (Date exacte ou +1 jour)
        games_date = games[games['GAME_DATE'] == bet['Date']]
        if games_date.empty:
             games_date = games[games['GAME_DATE'] == bet['Date'] + pd.Timedelta(days=1)]

        found_row = None
        
        for _, row in games_date.iterrows():
            matchup_str = row['MATCHUP'].replace('@', ' ').replace('vs.', ' ')
            # On vérifie si l'abbréviation de notre équipe est dans le texte du matchup
            if pred_abbr in matchup_str:
                # On vérifie que ce n'est pas un faux positif (ex: "NYK" dans "NYK" ok)
                # On regarde la ligne spécifique de l'équipe
                # Si la ligne contient l'abbréviation dans ses valeurs (souvent colonne 2 ou 3)
                row_str = str(row.values)
                if f"'{pred_abbr}'" in row_str or f" {pred_abbr} " in row_str:
                    found_row = row
                    break
        
        if found_row is not None:
            res = found_row['WL'] # W ou L
            
            # Qui a gagné ?
            is_win = (res == 'W')
            
            if is_win:
                real_winner_name = pred_full
            else:
                # Si j'ai parié Home et perdu, c'est Away qui a gagné
                real_winner_name = bet['Away'] if pred_full == bet['Home'] else bet['Home']

            status = "GAGNE" if is_win else "PERDU"
            
            bets.at[index, 'Result'] = status
            bets.at[index, 'Real_Winner'] = real_winner_name
            
            print(f"[MAJ] {pred_abbr}: {status} (Vainqueur: {real_winner_name})")
            updated_count += 1

if updated_count > 0:
    bets.to_csv(HISTORY_FILE, index=False)
    print(f"[OK] {updated_count} lignes corrigees.")
else:
    print("Tout est deja a jour.")