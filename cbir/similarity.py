"""
Calcul de similarité entre vecteurs et classement Top-K.
"""
import numpy as np


def calculer_similarites_cosinus(vecteur_requete, matrice_catalogue):
    """
    Retourne un score de similarité cosinus (entre -1 et 1) pour chaque
    ligne de matrice_catalogue, de forme (N, 2048).
    """
    produits_scalaires = matrice_catalogue @ vecteur_requete
    norme_requete = np.linalg.norm(vecteur_requete)
    normes_catalogue = np.linalg.norm(matrice_catalogue, axis=1)

    # 1e-12 évite une division par zéro si un vecteur est nul
    return produits_scalaires / (norme_requete * normes_catalogue + 1e-12)


def classer_top_k(scores, ids_produits, top_k=10):
    """
    Trie les scores du plus grand au plus petit et retourne les K meilleurs
    sous la forme [(product_id, score), ...].
    """
    if len(scores) == 0:
        return []   # catalogue vide

    top_k = min(top_k, len(scores))   # on ne peut pas retourner plus que le catalogue
    ordre = np.argsort(scores)[::-1][:top_k]

    return [(ids_produits[i], float(scores[i])) for i in ordre]