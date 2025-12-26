from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

# DATE CIBLE : 23 Décembre 2025 (Format US pour le Finder)
DATE_US = "12/23/2025" 

print(f"\n--- 🕵️‍♂️ DEBUGGING LEAGUE GAME FINDER : {DATE_US} ---\n")

try:
    print("Appel API en cours...")
    # On demande les matchs pour cette date précise
    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=DATE_US,
        date_to_nullable=DATE_US,
        league_id_nullable='00' # NBA
    )
    
    # CORRECTION ICI : get_data_frames() renvoie une liste, on prend le premier [0]
    results = finder.get_data_frames()[0]
    
    if results.empty:
        print("❌ FINDER VIDE (Aucun résultat retourné). L'API bloque peut-être cette date.")
    else:
        print(f"✅ {len(results)} Lignes de stats trouvées !")
        print("-" * 30)
        
        # On affiche les 2 premières lignes pour voir le format exact des IDs
        for i in range(min(2, len(results))):
            row = results.iloc[i]
            print(f"MATCH {i+1}: {row['MATCHUP']}")
            print(f"   ID Match (GAME_ID) : '{row['GAME_ID']}' (Type: {type(row['GAME_ID'])})")
            print(f"   ID Team (TEAM_ID)  : '{row['TEAM_ID']}' (Type: {type(row['TEAM_ID'])})")
            print(f"   Points (PTS)       : {row['PTS']}")
            print("-" * 30)
            
        print("\n💡 CONCLUSION POUR APP.PY :")
        print("Si tu vois des scores ci-dessus, on a la solution.")
        print("Il suffira de copier la logique de récupération de cet ID.")

except Exception as e:
    print(f"CRASH TEST : {e}")

input("\nAppuie sur Entrée pour quitter...")