import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n{'='*40}")
    print(f"🚀 LANCEMENT DE : {script_name}")
    print(f"{'='*40}\n")
    
    try:
        # On lance le script et on attend qu'il finisse
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ {script_name} terminé.")
    except subprocess.CalledProcessError:
        print(f"\n❌ Le script {script_name} a rencontré une erreur.")
        print("Arrêt de la séquence.")
        sys.exit(1)

# --- SÉQUENCE LOCALE ---

print("--- ROUTINE NBA (LOCALE) ---")

# 1. Mise à jour des données
run_script('data_nba.py')

# 2. Calculs
run_script('features_nba.py')

# 3. Prédictions (Hybrides)
# Tente l'auto, sinon passe en manuel
run_script('predict_nba.py')