import os
import requests
from nba_api.stats.static import teams
import time

# Configuration
LOGO_DIR = "logos"

print(f"--- TÉLÉCHARGEMENT DES LOGOS NBA ---")

# 1. Création du dossier s'il n'existe pas
if not os.path.exists(LOGO_DIR):
    os.makedirs(LOGO_DIR)
    print(f"📂 Dossier '{LOGO_DIR}' créé.")
else:
    print(f"📂 Dossier '{LOGO_DIR}' existant détecté.")

# 2. Récupération de la liste des équipes
nba_teams = teams.get_teams()
print(f"🎯 {len(nba_teams)} équipes trouvées.")

# 3. Boucle de téléchargement
count = 0
for team in nba_teams:
    team_id = team['id']
    abbrev = team['abbreviation']
    
    # URL officielle des logos NBA (Format SVG, très léger et net)
    url = f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
    
    # Nom du fichier local (ex: logos/1610612747.svg)
    filename = f"{LOGO_DIR}/{team_id}.svg"
    
    # On ne télécharge que si on ne l'a pas déjà
    if not os.path.exists(filename):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✅ {abbrev} téléchargé.")
                count += 1
            else:
                print(f"❌ {abbrev} introuvable (Code {response.status_code})")
        except Exception as e:
            print(f"⚠️ Erreur pour {abbrev} : {e}")
        
        # Petite pause pour être poli avec le serveur
        time.sleep(0.2)
    else:
        print(f"➡️ {abbrev} déjà présent.")

print(f"\n✨ Terminé ! {count} nouveaux logos récupérés dans le dossier '{LOGO_DIR}'.")