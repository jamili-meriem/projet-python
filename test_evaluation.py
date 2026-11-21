from pathlib import Path
from cbir.cbir_engine import search_similar

# --- Test 1 : images hors catalogue ---
print("=== Images hors catalogue ===")
for chemin in sorted(Path("tests_images").glob("*")):
    resultats = search_similar(chemin, top_k=3)
    affichage = [(pid, round(score, 3)) for pid, score in resultats]
    print(chemin.name, "->", affichage)

# --- Test 2 : cas d'erreur ---
print("\n=== Cas d'erreur ===")
for chemin in ["dataset/faux.jpg", "dataset/n_existe_pas.jpg"]:
    try:
        search_similar(chemin)
        print(chemin, "-> aucune erreur (?)")
    except Exception as erreur:
        print(chemin, "-> erreur attrapée :", type(erreur).__name__)


# --- Precision@K (prêt pour le vrai catalogue) ---
def precision_at_k(resultats, categorie_requete, categorie_par_id):
    """Part des K résultats qui sont de la même catégorie que la requête."""
    if not resultats:
        return 0.0
    bons = sum(1 for pid, _ in resultats
               if categorie_par_id.get(str(pid)) == categorie_requete)
    return bons / len(resultats)