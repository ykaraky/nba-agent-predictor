import subprocess
import sys
import time

def run_script(script_name):
    print(f"\n{'='*40}")
    print(f"🚀 LANCEMENT DE : {script_name}")
    print(f"{'='*40}\n")
    
    # sys.executable assure qu'on utilise le même Python que celui en cours
    # check=True permet d'arrêter tout si un script plante (ex: pas d'internet)
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ {script_name} terminé avec succès.")
    except subprocess.CalledProcessError:
        print(f"\n❌ ERREUR CRITIQUE dans {script_name}.")
        print("Arrêt du programme.")
        sys.exit(1) # On quitte tout

# --- DÉBUT DU PIPELINE ---

print("--- MISE À JOUR QUOTIDIENNE DE L'AGENT NBA ---")

# Étape 1 : Récupérer les nouveaux matchs de la nuit
run_script('data_nba.py')

# Étape 2 : Recalculer les moyennes et la fatigue
run_script('features_nba.py')

# Pause courte pour être sûr que les fichiers sont bien enregistrés sur le disque
time.sleep(1)

# Étape 3 : Lancer l'interface de prédiction (le script manuel qui marche bien)
# Note : On ne relance pas l'entraînement (train_nba.py) tous les jours, 
# ce n'est pas nécessaire et c'est long.
run_script('predict_manual.py')