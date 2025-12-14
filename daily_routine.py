import subprocess
import sys
import time
import os
from datetime import datetime

def run_step(script_name, description):
    print(f"\n{'='*50}")
    print(f"🚀 ÉTAPE : {description}")
    print(f"{'='*50}")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"✅ {script_name} terminé avec succès.")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ ERREUR CRITIQUE dans {script_name}.")
        return False

def run_git_sync():
    print(f"\n{'='*50}")
    print(f"☁️ SYNCHRONISATION CLOUD (GITHUB)")
    print(f"{'='*50}")
    try:
        # 1. Add
        subprocess.run(["git", "add", "."], check=True)
        
        # 2. Commit avec la date
        date_msg = datetime.now().strftime('%Y-%m-%d %H:%M')
        commit_msg = f"Auto-update scores & predictions {date_msg}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False) # check=False car si rien à commiter, ça renvoie une erreur code 1
        
        # 3. Push
        print("Envoi vers GitHub...")
        subprocess.run(["git", "push"], check=True)
        print("✅ Synchro terminée ! Ton site mobile est à jour.")
    except Exception as e:
        print(f"⚠️ Erreur Git (pas grave si c'est juste réseau) : {e}")

# --- DÉMARRAGE DU PROTOCOLE ---

print("\n🏀 --- NBA AGENT : ROUTINE MATINALE AUTOMATISÉE --- 🏀\n")

# 1. Téléchargement des scores d'hier
if not run_step('data_nba.py', "Mise à jour des Scores"):
    input("Appuie sur Entrée pour quitter...")
    exit()

# 2. Recalcul des stats (Four Factors)
run_step('features_nba.py', "Recalcul des Statistiques")

# 3. Vérification des paris d'hier (GAGNÉ/PERDU)
run_step('verify_bets.py', "Validation des résultats d'hier")

# 4. Génération des pronostics pour ce soir (NOUVEAU)
run_step('predict_today.py', "Génération des Pronostics du jour")

# 5. Envoi sur le Cloud
run_git_sync()

# 6. Ouverture de l'interface pour voir le résultat
print("\n✨ Tout est prêt. Lancement de l'interface...")
time.sleep(2)
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])