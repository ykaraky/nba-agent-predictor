import subprocess
import sys
import time
import os
from datetime import datetime

def run_step(script_path, description):
    print(f"\n{'='*50}")
    print(f"🚀 ÉTAPE : {description}")
    print(f"{'='*50}")
    
    # On vérifie que le fichier existe avant de lancer
    if not os.path.exists(script_path):
        print(f"❌ ERREUR : Le fichier {script_path} est introuvable.")
        return False

    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ {script_path} terminé avec succès.")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ ERREUR CRITIQUE dans {script_path}.")
        return False

def run_git_sync():
    print(f"\n{'='*50}")
    print(f"☁️ SYNCHRONISATION CLOUD")
    print(f"{'='*50}")
    try:
        subprocess.run(["git", "add", "."], check=True)
        date_msg = datetime.now().strftime('%Y-%m-%d %H:%M')
        subprocess.run(["git", "commit", "-m", f"Auto-update v5 {date_msg}"], check=False)
        print("Envoi vers GitHub...")
        subprocess.run(["git", "push"], check=True)
        print("✅ Synchro terminée !")
    except Exception as e:
        print(f"⚠️ Erreur Git : {e}")

# --- DÉMARRAGE v5 ---

print("\n🏀 --- NBA AGENT v5 : ROUTINE --- 🏀\n")

# Note les chemins : src/nom_du_fichier.py
if not run_step('src/data_nba.py', "Mise à jour des Scores"):
    input("Entrée pour quitter...")
    exit()

run_step('src/features_nba.py', "Recalcul Stats")
run_step('src/verify_bets.py', "Vérification Paris")

# Envoi Cloud
run_git_sync()

print("\n✨ Lancement de l'interface v5...")
time.sleep(2)
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])