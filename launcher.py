import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n{'='*40}")
    print(f"🚀 LANCEMENT DE : {script_name}")
    print(f"{'='*40}\n")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ {script_name} terminé.")
    except subprocess.CalledProcessError:
        print(f"\n❌ Erreur dans {script_name}. Arrêt.")
        sys.exit(1)

print("--- ROUTINE NBA ---")

# 1. Mise à jour des données (Récupère les scores d'hier)
run_script('data_nba.py')

# 2. Calculs stats
run_script('features_nba.py')

# 3. VÉRIFICATION DES RÉSULTATS (NOUVEAU !)
# On regarde si nos paris d'hier étaient bons
run_script('verify_bets.py')

# Pause lecture
time.sleep(2)

# 4. Prédictions pour aujourd'hui
run_script('predict_nba.py')