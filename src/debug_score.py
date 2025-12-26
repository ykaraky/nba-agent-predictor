from nba_api.stats.endpoints import scoreboardv2, leaguegamefinder
import pandas as pd
from datetime import datetime

# DATE CIBLE : 23 Décembre 2025
DATE_ISO = "2025-12-23" # Pour Scoreboard
DATE_US = "12/23/2025"  # Pour Finder

print(f"\n--- 🕵️‍♂️ DEBUGGING SCORES : {DATE_ISO} ---\n")

# --- TEST 1 : SCOREBOARD V2 ---
print("1️⃣ TEST SCOREBOARD V2 (Squelette)")
try:
    board = scoreboardv2.ScoreboardV2(game_date=DATE_ISO)
    header = board.game_header.get_data_frame()
    lines = board.line_score.get_data_frame()
    
    if header.empty:
        print("❌ HEADER VIDE (Pas de matchs trouvés)")
    else:
        print(f"✅ {len(header)} Matchs trouvés.")
        # Affichons un ID brut pour voir le format
        sample_id = header.iloc[0]['GAME_ID']
        print(f"   Exemple ID Match (Header) : '{sample_id}' (Type: {type(sample_id)})")
    
    if lines.empty:
        print("❌ LINE_SCORE VIDE (L'API ne renvoie pas les scores ici)")
    else:
        print(f"✅ {len(lines)} Scores trouvés.")
        sample_id_score = lines.iloc[0]['GAME_ID']
        print(f"   Exemple ID Score (Line)   : '{sample_id_score}' (Type: {type(sample_id_score)})")

except Exception as e:
    print(f"CRASH TEST 1: {e}")

print("\n" + "="*30 + "\n")

# --- TEST 2 : LEAGUE GAME FINDER (Muscles) ---
print("2️⃣ TEST LEAGUE GAME FINDER (Détails)")
try:
    # On demande les matchs pour cette date précise
    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=DATE_US,
        date_to_nullable=DATE_US,
        league_id_nullable='00' # NBA
    )
    results = finder.get_data_frame()
    
    if results.empty:
        print("❌ FINDER VIDE (Aucun résultat retourné)")
    else:
        print(f"✅ {len(results)} Lignes trouvées.")
        
        # Affichons les colonnes intéressantes pour le premier résultat
        first_row = results.iloc[0]
        print(f"   Exemple ID Match : '{first_row['GAME_ID']}' (Type: {type(first_row['GAME_ID'])})")
        print(f"   Exemple Team ID  : '{first_row['TEAM_ID']}'")
        print(f"   Exemple PTS      : {first_row['PTS']}")
        print(f"   Exemple Matchup  : {first_row['MATCHUP']}")
        
        # Test de matching
        print("\n   🔍 TEST DE MATCHING :")
        # On essaie de convertir en int pour voir
        try:
            raw_id = first_row['GAME_ID']
            clean_id = str(int(float(raw_id)))
            print(f"   Conversion '{raw_id}' -> INT -> STR : '{clean_id}'")
        except:
            print("   Echec de la conversion ID.")

except Exception as e:
    print(f"CRASH TEST 2: {e}")

print("\n------------------------------------------------")