import numpy as np

# --- Le "catalogue" : 4 produits, chacun décrit par 3 nombres ---
# (en vrai : 2048 nombres par produit)
catalogue = np.array([
    [1.0, 2.0, 0.0],   # produit 101 : sneaker A
    [2.0, 3.5, 0.5],   # produit 102 : sneaker B
    [0.0, 0.5, 3.0],   # produit 103 : cafetière
    [4.0, 2.0, 1.0],   # produit 104 : chaussure de ville
])
ids_produits = np.array([101, 102, 103, 104])

# --- L'image envoyée par l'utilisateur (déjà transformée en vecteur) ---
requete = np.array([1.0, 2.0, 0.0])


def similarite_cosinus(vecteur_requete, matrice_catalogue):
    """Retourne un score de similarité cosinus pour chaque ligne du catalogue."""
    # Produit scalaire entre la requête et chaque produit (tout d'un coup)
    produits_scalaires = matrice_catalogue @ vecteur_requete
    # Longueur (norme) de la requête
    norme_requete = np.linalg.norm(vecteur_requete)
    # Longueur (norme) de chaque produit du catalogue
    normes_catalogue = np.linalg.norm(matrice_catalogue, axis=1)
    # Formule du cosinus
    return produits_scalaires / (norme_requete * normes_catalogue)


scores = similarite_cosinus(requete, catalogue)
print("Scores :", scores.round(4))

# --- Classement Top-K ---
top_k = 3
ordre = np.argsort(scores)[::-1]   # positions triées du meilleur au moins bon

print(f"\nTop {top_k} :")
for position in ordre[:top_k]:
    print("produit", ids_produits[position], "-> score", round(float(scores[position]), 4))