import numpy as np
from cbir.feature_extraction import charger_modele, extraire_vecteur
from cbir.similarity import calculer_similarites_cosinus, classer_top_k

# Catalogue pré-calculé (chargé sans recalcul)
vecteurs = np.load("features/product_vectors.npy")
ids = np.load("features/product_ids.npy")

# Image requête : on prend a1, qui est dans le catalogue
modele = charger_modele()
requete = extraire_vecteur("dataset/a1.jpg", modele)

scores = calculer_similarites_cosinus(requete, vecteurs)
resultats = classer_top_k(scores, ids, top_k=3)

for product_id, score in resultats:
    print(product_id, "->", round(score, 3))