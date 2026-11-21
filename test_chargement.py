import numpy as np

vecteurs = np.load("features/product_vectors.npy")
ids = np.load("features/product_ids.npy")

print("Vecteurs :", vecteurs.shape, vecteurs.dtype)
print("Ids :", ids)