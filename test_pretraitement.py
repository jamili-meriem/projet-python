import numpy as np
from cbir.feature_extraction import pretraiter_image

# Remplace par le nom exact de ta photo
chemin = "dataset/test1.jpg"

image_prete = pretraiter_image(chemin)

print("Forme :", image_prete.shape)
print("Type :", image_prete.dtype)
print("Minimum :", image_prete.min())
print("Maximum :", image_prete.max())