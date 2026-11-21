"""
Extraction des caractéristiques visuelles (moteur CBIR).

Contenu actuel :
  - charger_modele()   : charge ResNet50 pré-entraîné, sans couche finale
  - pretraiter_image() : prépare une image pour ResNet50

Règle essentielle : le MÊME modèle et le MÊME prétraitement doivent être
utilisés pour les images du catalogue et pour l'image de l'utilisateur.
"""
import os

# Réduit les messages d'information de TensorFlow dans le terminal.
# Doit être placé AVANT l'import de Keras/TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

# Taille d'entrée imposée par ResNet50
TAILLE_IMAGE = (224, 224)


def charger_modele():
    """
    Charge ResNet50 pré-entraîné sur ImageNet, SANS la couche de
    classification finale. Le modèle retourné transforme une image
    en un vecteur de 2048 nombres.
    """
    modele = ResNet50(
        weights="imagenet",          # poids déjà appris sur ImageNet
        include_top=False,           # on retire la couche de classification
        pooling="avg",               # résume la sortie en un vecteur de 2048 nombres
        input_shape=(224, 224, 3),   # images 224 x 224 pixels, 3 canaux (RGB)
    )
    return modele


def pretraiter_image(chemin_image):
    """
    Lit une image et la prépare pour ResNet50.
    Retourne un tableau NumPy de forme (1, 224, 224, 3).
    Lève une exception si le fichier n'est pas une image exploitable.
    """
    # 1. Ouvrir l'image et vérifier qu'elle n'est pas corrompue
    with Image.open(chemin_image) as image:
        image.verify()

    # verify() rend l'image inutilisable : on la rouvre pour la traiter
    with Image.open(chemin_image) as image:
        # 2. Convertir en RGB (gère niveaux de gris, PNG transparents...)
        image = image.convert("RGB")
        # 3. Redimensionner à 224 x 224
        image = image.resize(TAILLE_IMAGE)
        # 4. Convertir en tableau de nombres
        tableau = np.array(image, dtype="float32")

    # 5. Ajouter la dimension "lot" : (224, 224, 3) -> (1, 224, 224, 3)
    tableau = np.expand_dims(tableau, axis=0)

    # 6. Normalisation propre à ResNet50
    return preprocess_input(tableau)
def extraire_vecteur(chemin_image, modele):
    """
    Transforme une image en vecteur de 2048 nombres (tableau NumPy 1D).
    Le modèle est passé en paramètre pour ne le charger qu'une seule fois.
    """
    image_prete = pretraiter_image(chemin_image)
    vecteur = modele.predict(image_prete, verbose=0)
    return vecteur[0]   # (1, 2048) -> (2048,)