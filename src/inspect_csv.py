import pandas as pd

CSV_PATH = 'data/bets_history.csv'

print("--- INSPECTION DES DATES CSV ---")
df = pd.read_csv(CSV_PATH)

# Afficher les 10 dernières dates uniques
print("\n10 dernières dates trouvées dans le fichier :")
print(df['Date'].unique()[-10:])

# Vérifier spécifiquement le 28
print("\nRecherche spécifique '2025-12-28' :")
matches = df[df['Date'].astype(str).str.contains("2025-12-28")]
if not matches.empty:
    print(f"✅ Trouvé ! {len(matches)} lignes.")
    print(matches[['Date', 'Home', 'Away']].head())
else:
    print("❌ Aucune ligne ne contient '2025-12-28'.")
    # Essayons avec le format européen
    matches_eu = df[df['Date'].astype(str).str.contains("28.12.2025")]
    if not matches_eu.empty:
        print(f"⚠️ Trouvé au format '28.12.2025' ! ({len(matches_eu)} lignes)")