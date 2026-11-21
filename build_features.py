"""
Pré-calcul des vecteurs du catalogue.
À lancer une seule fois, puis à chaque changement d'images.
Crée : features/product_vectors.npy et features/product_ids.npy
"""
from pathlib import Path
import numpy as np
from cbir.feature_extraction import charger_modele, extraire_vecteur

DOSSIER_DATASET = Path("dataset")
DOSSIER_FEATURES = Path("features")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def construire_catalogue():
    modele = charger_modele()   # chargé une seule fois

    # Toutes les images, y compris dans les sous-dossiers (catégories)
    chemins = sorted(
        p for p in DOSSIER_DATASET.rglob("*") if p.suffix.lower() in EXTENSIONS
    )
    print(f"{len(chemins)} images trouvées")

    vecteurs, ids, erreurs = [], [], []
    for numero, chemin in enumerate(chemins, start=1):
        try:
            vecteur = extraire_vecteur(str(chemin), modele)
        except Exception as erreur:
            # Image illisible : on la saute, on continue avec les autres
            erreurs.append((chemin, erreur))
            print(f"[{numero}/{len(chemins)}] ERREUR {chemin} : {erreur}")
            continue
        vecteurs.append(vecteur)
        ids.append(chemin.stem)   # nom du fichier sans extension = product_id
        print(f"[{numero}/{len(chemins)}] OK {chemin}")

    if len(set(ids)) != len(ids):
        print("ATTENTION : des identifiants sont en double (même nom de fichier)")

    DOSSIER_FEATURES.mkdir(exist_ok=True)
    np.save(DOSSIER_FEATURES / "product_vectors.npy", np.array(vecteurs, dtype="float32"))
    np.save(DOSSIER_FEATURES / "product_ids.npy", np.array(ids))

    print(f"\nTerminé : {len(vecteurs)} vecteurs sauvegardés, {len(erreurs)} erreur(s)")


if __name__ == "__main__":
    construire_catalogue()