import numpy as np
from cbir.feature_extraction import charger_modele

modele = charger_modele()

print("Entrée attendue :", modele.input_shape)
print("Sortie produite :", modele.output_shape)

# Fausse image (du bruit aléatoire), juste pour vérifier la forme de la sortie
fausse_image = np.random.rand(1, 224, 224, 3).astype("float32") * 255
vecteur = modele.predict(fausse_image, verbose=0)

print("Forme du vecteur :", vecteur.shape)
print("5 premiers nombres :", vecteur[0][:5])