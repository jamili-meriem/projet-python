"""
Moteur CBIR : fonction principale search_similar(image_path, top_k=10).
Contrat avec Flask : retourne [(product_id, similarity_score), ...]
trié du plus similaire au moins similaire.
"""
from pathlib import Path
import numpy as np

from cbir.feature_extraction import charger_modele, extraire_vecteur
from cbir.similarity import calculer_similarites_cosinus, classer_top_k

# Chemins calculés depuis l'emplacement de ce fichier : ça marche
# quelle que soit l'endroit d'où Flask est lancé.
RACINE_PROJET = Path(__file__).resolve().parent.parent
FICHIER_VECTEURS = RACINE_PROJET / "features" / "product_vectors.npy"
FICHIER_IDS = RACINE_PROJET / "features" / "product_ids.npy"

# Mémoire : chargés une seule fois, à la première recherche
_modele = None
_vecteurs = None
_ids = None


def _charger_ressources():
    """Charge le modèle et les vecteurs du catalogue s'ils ne le sont pas déjà."""
    global _modele, _vecteurs, _ids
    if _modele is None:
        _modele = charger_modele()
    if _vecteurs is None:
        if not FICHIER_VECTEURS.exists() or not FICHIER_IDS.exists():
            raise FileNotFoundError(
                "Vecteurs introuvables : lance d'abord build_features.py"
            )
        _vecteurs = np.load(FICHIER_VECTEURS)
        _ids = np.load(FICHIER_IDS)


def _convertir_id(identifiant):
    """'12' -> 12 (entier) si possible, car SQLite utilise des id entiers."""
    texte = str(identifiant)
    return int(texte) if texte.isdigit() else texte


def search_similar(image_path, top_k=10):
    """
    Cherche les produits visuellement similaires à l'image donnée.

    Entrée  : image_path = chemin local de l'image (str ou Path)
              top_k      = nombre maximum de résultats
    Sortie  : [(product_id, similarity_score), ...], du plus similaire
              au moins similaire ([] si le catalogue est vide)
    Erreurs : FileNotFoundError / PIL.UnidentifiedImageError si l'image
              est introuvable ou illisible (à attraper côté Flask).
    """
    _charger_ressources()

    vecteur_requete = extraire_vecteur(str(image_path), _modele)
    scores = calculer_similarites_cosinus(vecteur_requete, _vecteurs)
    resultats = classer_top_k(scores, _ids, top_k)

    return [(_convertir_id(pid), score) for pid, score in resultats]