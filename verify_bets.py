import pandas as pd
import os

HISTORY_FILE = 'bets_history.csv'
GAMES_FILE = 'nba_games.csv'

print(f"\n🔎 --- VÉRIFICATION DES PARIS ---")

if not os.path.exists(HISTORY_FILE):
    print("Pas d'historique de paris trouvé pour l'instant.")
    exit()

# 1. Charger l'historique des paris
bets = pd.read_csv(HISTORY_FILE)
bets['Date'] = pd.to_datetime(bets['Date'])

# 2. Charger les résultats réels (mis à jour ce matin par data_nba.py)
games = pd.read_csv(GAMES_FILE)
games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])

# On filtre pour ne garder que les matchs récents pour aller plus vite
games = games[games['GAME_DATE'] >= '2023-01-01']

updated_count = 0
wins = 0
losses = 0

# 3. Boucle de vérification
# On parcourt chaque pari qui n'a pas encore de résultat (colonne Result vide ou NaN)
for index, bet in bets.iterrows():
    if pd.isna(bet['Result']) or bet['Result'] == '':
        
        # On cherche le match dans la base de données réelle
        # Critères : Même date et l'équipe à domicile correspond
        match_reel = games[
            (games['GAME_DATE'] == bet['Date']) & 
            (games['MATCHUP'].str.contains(bet['Home'])) & # Contient le nom de l'équipe Home
            (games['MATCHUP'].str.contains('vs.')) # C'est bien un match à domicile
        ]
        
        if len(match_reel) > 0:
            # Le match a été joué !
            real_result = match_reel.iloc[0]
            
            # Qui a gagné en vrai ?
            real_winner = bet['Home'] if real_result['WL'] == 'W' else bet['Away']
            
            # Verdict
            status = "GAGNÉ" if bet['Predicted_Winner'] == real_winner else "PERDU"
            
            # Mise à jour
            bets.at[index, 'Result'] = status
            print(f"📝 Match du {bet['Date'].date()} ({bet['Home']} vs {bet['Away']}) : {status}")
            updated_count += 1
        else:
            # Match pas encore joué ou données pas encore dispos
            pass

# 4. Sauvegarde des modifications
if updated_count > 0:
    bets.to_csv(HISTORY_FILE, index=False)
    print(f"\n✅ {updated_count} paris mis à jour dans l'historique.")
else:
    print("Aucun nouveau résultat trouvé.")

# 5. Statistiques Globales
completed_bets = bets.dropna(subset=['Result'])
total_completed = len(completed_bets)

if total_completed > 0:
    nb_gagnes = len(completed_bets[completed_bets['Result'] == 'GAGNÉ'])
    accuracy = (nb_gagnes / total_completed) * 100
    
    print("\n📊 --- BILAN DE L'AGENT ---")
    print(f"Total Paris : {total_completed}")
    print(f"Victoires   : {nb_gagnes}")
    print(f"Défaites    : {total_completed - nb_gagnes}")
    print(f"PRÉCISION   : {accuracy:.1f}%")
    
    if accuracy > 55:
        print("🔥 L'agent est RENTABLE !")
    elif accuracy > 50:
        print("⚖️ L'agent est à l'équilibre.")
    else:
        print("❄️ L'agent perd de l'argent.")
else:
    print("\nPas encore assez de données pour les statistiques.")