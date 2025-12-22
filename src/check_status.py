import sys
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2
import pandas as pd

def check_nba_status():
    # 1. Date cible : Hier
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y-%m-%d')
    date_disp = yesterday.strftime('%d.%m.%Y')
    
    print(f"--- DIAGNOSTIC NBA API ({date_disp}) ---")
    print("Connexion API en cours...")

    try:
        # 2. Interrogation Scoreboard
        board = scoreboardv2.ScoreboardV2(game_date=date_str)
        games = board.game_header.get_data_frame()
        line_scores = board.line_score.get_data_frame()

        if games.empty:
            print(f"[INFO] Aucun match trouve pour la date du {date_disp}.")
            print(">> Tu peux lancer la mise a jour (rien ne changera).")
            return

        total_games = len(games)
        ready_count = 0
        
        print(f"\n{total_games} matchs trouves hier :")
        print("-" * 40)

        # 3. Analyse détaillée match par match
        for i, game in games.iterrows():
            game_id = game['GAME_ID']
            status_id = game['GAME_STATUS_ID'] # 3 = Final
            status_text = str(game['GAME_STATUS_TEXT']).strip()
            
            # On recupere les codes equipes
            home_id = game['HOME_TEAM_ID']
            away_id = game['VISITOR_TEAM_ID']
            
            # Recup scores (si dispo)
            pts_home = 0
            pts_away = 0
            
            if not line_scores.empty:
                ls_home = line_scores[line_scores['TEAM_ID'] == home_id]
                ls_away = line_scores[line_scores['TEAM_ID'] == away_id]
                # On cherche la ligne correspondant a ce game_id
                ls_home = ls_home[ls_home['GAME_ID'] == game_id]
                ls_away = ls_away[ls_away['GAME_ID'] == game_id]
                
                if not ls_home.empty and 'PTS' in ls_home.columns:
                    pts_home = ls_home.iloc[0]['PTS']
                if not ls_away.empty and 'PTS' in ls_away.columns:
                    pts_away = ls_away.iloc[0]['PTS']

            # LOGIQUE DE VALIDATION SOUPLE
            # Est considere pret si : Statut est Final OU (Scores > 0 et Statut contient Final)
            is_final_status = (status_id == 3)
            has_scores = (pd.notna(pts_home) and int(pts_home) > 0 and pd.notna(pts_away) and int(pts_away) > 0)
            
            is_ready = is_final_status or has_scores
            
            # Affichage ligne
            match_label = f"Match {i+1}"
            score_display = f"{int(pts_away)} - {int(pts_home)}" if has_scores else "? - ?"
            state_display = "[OK]" if is_ready else "[EN COURS]"
            
            print(f"{state_display} {match_label} : {score_display} ({status_text})")
            
            if is_ready:
                ready_count += 1

        print("-" * 40)
        print(f"BILAN : {ready_count} / {total_games} matchs prets.")

        # 4. Verdict
        if ready_count == total_games and total_games > 0:
            print("\n>>> FEU VERT : GO_NBA.bat AUTORISE <<<")
        elif ready_count == 0:
            print("\n>>> FEU ROUGE : RIEN N'EST PRET <<<")
        else:
            print(f"\n>>> FEU ORANGE : Manque {total_games - ready_count} resultats <<<")
            print("Si les scores affiches ci-dessus semblent complets, tu peux tenter.")

    except Exception as e:
        print(f"[ERREUR] Probleme technique API : {e}")

if __name__ == "__main__":
    check_nba_status()