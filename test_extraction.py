import numpy as np
from cbir.feature_extraction import charger_modele, extraire_vecteur

modele = charger_modele()

v_a1 = extraire_vecteur("dataset/a1.jpg", modele)
v_a2 = extraire_vecteur("dataset/a2.jpg", modele)
v_b1 = extraire_vecteur("dataset/b1.jpg", modele)

print("Forme :", v_a1.shape)

def cosinus(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

print("a1 vs a2 (même type)  :", round(cosinus(v_a1, v_a2), 3))
print("a1 vs b1 (différents) :", round(cosinus(v_a1, v_b1), 3))
print("a1 vs a1 (identique)  :", round(cosinus(v_a1, v_a1), 3))