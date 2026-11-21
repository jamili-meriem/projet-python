# Application Web E-commerce avec Recherche d'Articles par Image (CBIR)

Mini-projet Python – Sciences des données – 2026–2027
Équipe : **Meriem Jamili** (IA et moteur CBIR) et **Marwa** (application web, back-end et base de données).

Ce document décrit la partie **IA et moteur CBIR**. La partie application web (Flask, SQLite, interface) est documentée par Marwa dans sa propre section.

---

## 1. Principe

L'utilisateur envoie la photo d'un article. Le moteur retourne les produits du catalogue qui lui ressemblent visuellement.

1. **Avant (une seule fois)** : chaque image du catalogue est prétraitée, passée dans ResNet50 (sans sa couche de classification) et transformée en un **vecteur de 2048 nombres**. Les vecteurs sont sauvegardés dans `features/`.
2. **À chaque recherche** : seule l'image de l'utilisateur est transformée en vecteur (même modèle, même prétraitement), puis comparée aux vecteurs sauvegardés avec la **similarité cosinus**. Les K meilleurs résultats sont retournés.

**Choix techniques**

| Élément | Choix | Raison |
|---|---|---|
| Modèle | ResNet50 pré-entraîné sur ImageNet (`include_top=False`, `pooling="avg"`) | Choix conseillé par le cahier des charges ; vecteurs riches de 2048 dimensions |
| Prétraitement | Pillow : conversion RGB, redimensionnement 224 × 224, `preprocess_input` de ResNet50 | Normalisation identique à celle de l'entraînement du modèle |
| Similarité | Cosinus (NumPy) | Compare l'orientation des vecteurs, score borné entre -1 et 1 |
| Classement | Tri décroissant, Top-K | Résultats du plus au moins similaire |

---

## 2. Structure du moteur

```
projet_cbir/
├── cbir/
│   ├── __init__.py            # rend le dossier importable
│   ├── feature_extraction.py  # chargement du modèle, prétraitement, extraction du vecteur
│   ├── similarity.py          # similarité cosinus et classement Top-K
│   └── cbir_engine.py         # fonction principale search_similar()
├── dataset/                   # images du catalogue (par catégories)
├── features/
│   ├── product_vectors.npy    # vecteurs pré-calculés
│   └── product_ids.npy        # identifiants produits correspondants
├── build_features.py          # script de pré-calcul des vecteurs
├── test_search.py             # test de la fonction search_similar
├── test_evaluation.py         # images hors catalogue, cas d'erreur, Precision@K
├── requirements.txt           # dépendances Python
└── README.md
```

Les autres fichiers `test_*.py` sont des tests d'étapes intermédiaires (modèle, prétraitement, extraction, similarité).

---

## 3. Installation (Windows, PowerShell)

Python 3.10 à 3.13 est supporté par TensorFlow 2.21 (développé et testé avec Python 3.13).

```powershell
cd projet_cbir
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell refuse d'exécuter le script d'activation, autoriser les scripts pour l'utilisateur courant avec `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Remarques**
- Au **premier lancement**, Keras télécharge les poids ImageNet de ResNet50 (environ 95 Mo). Une connexion Internet est nécessaire une seule fois ; les poids sont ensuite conservés dans `C:\Users\<utilisateur>\.keras\models`.
- TensorFlow ≥ 2.11 n'utilise pas le GPU sous Windows natif : le calcul se fait sur le CPU, ce qui est suffisant pour ce projet.
- Les messages `oneDNN` et `CPU instructions` affichés par TensorFlow sont des informations, pas des erreurs.

---

## 4. Pré-calcul des vecteurs du catalogue

**Convention d'identifiants : le nom du fichier image (sans extension) est l'identifiant du produit.**
Exemple : `dataset/chaussures/12.jpg` correspond au produit `12` de la base SQLite. Deux images ne doivent pas porter le même nom, même dans des dossiers différents.

Lancer depuis la racine du projet :

```powershell
python build_features.py
```

Le script parcourt `dataset/` (y compris les sous-dossiers), extrait un vecteur par image, **ignore les images illisibles** (en affichant un message) et sauvegarde les résultats.

**À relancer** chaque fois que les images du catalogue changent. Il faut ensuite **redémarrer l'application Flask**, car les vecteurs sont gardés en mémoire.

### Format des fichiers sauvegardés

| Fichier | Contenu | Forme | Type |
|---|---|---|---|
| `features/product_vectors.npy` | Un vecteur par image, 1 ligne = 1 produit | `(N, 2048)` | `float32` |
| `features/product_ids.npy` | Identifiant du produit de chaque ligne | `(N,)` | chaînes de caractères |

La ligne `i` de `product_vectors.npy` correspond à l'identifiant à la position `i` de `product_ids.npy`. Les deux fichiers doivent toujours être régénérés ensemble.

Rechargement sans recalcul :

```python
import numpy as np
vecteurs = np.load("features/product_vectors.npy")
ids = np.load("features/product_ids.npy")
```

---

## 5. Utilisation : `search_similar`

```python
from cbir.cbir_engine import search_similar, preparer_moteur

preparer_moteur()   # optionnel : charge le modèle et les vecteurs au démarrage

resultats = search_similar("chemin/vers/image.jpg", top_k=10, score_min=0.40)
# [(12, 0.83), (7, 0.71), (31, 0.55)]
```

| Élément | Description |
|---|---|
| **Entrée** | `image_path` : chemin local de l'image téléversée (`str` ou `Path`) ; `top_k` : nombre maximum de résultats (10 par défaut) ; `score_min` : score minimum facultatif |
| **Sortie** | Liste `[(product_id, similarity_score), ...]` triée du plus similaire au moins similaire |
| `product_id` | Entier si le nom du fichier est numérique (`"12"` devient `12`), sinon texte |
| `similarity_score` | Nombre décimal entre -1 et 1 ; plus il est élevé, plus l'image est proche |
| **Liste vide** | Catalogue vide, ou aucun produit n'atteint `score_min` : afficher « aucun résultat pertinent » |
| **Erreurs** | `FileNotFoundError` (image ou vecteurs introuvables), `PIL.UnidentifiedImageError` (fichier non image) : à attraper dans Flask avec `try/except` |

**Performance** : le modèle et les vecteurs sont chargés une seule fois (cache en mémoire). Mesures sur le CPU de développement : environ 5 s pour la première recherche (chargement), environ 0,25 s pour les suivantes.

### Seuil de pertinence (`score_min`)
Sur les trois images de test, les correspondances pertinentes obtiennent un score d'environ 0,47 ou plus, et une image sans rapport reste sous 0,35. Un seuil **provisoire de 0,40** est proposé ; il doit être recalibré avec le catalogue complet.

---

## 6. Intégration avec Flask

```python
from cbir.cbir_engine import search_similar, preparer_moteur

preparer_moteur()   # au démarrage de l'application

# Dans la route de recherche, après avoir enregistré l'image téléversée :
try:
    resultats = search_similar(chemin_temporaire, top_k=10, score_min=0.40)
except Exception:
    ...  # fichier illisible ou vecteurs absents : afficher un message d'erreur

ids = [product_id for product_id, score in resultats]
# SELECT * FROM products WHERE id IN (...)
# puis réordonner les produits selon l'ordre de `ids`
```

**Points d'attention**
- Un `WHERE id IN (...)` **ne conserve pas l'ordre** : il faut réordonner les produits selon la liste retournée, sinon le classement par similarité est perdu.
- Flask doit être lancé depuis la racine du projet (là où se trouve `cbir/`) pour que `from cbir.cbir_engine import ...` fonctionne. Les chemins vers `features/` sont calculés depuis l'emplacement du module.
- Le contrat d'intégration est : `search_similar(image_path, top_k=10)` retourne `[(product_id, similarity_score), ...]`.

---

## 7. Tests

Depuis la racine du projet, environnement activé :

```powershell
python test_search.py        # recherche + temps de la 1re et de la 2e recherche
python test_evaluation.py    # images hors catalogue (dossier tests_images/) et cas d'erreur
```

`test_evaluation.py` contient aussi la fonction `precision_at_k`, destinée à l'évaluation sur le catalogue complet (la catégorie d'un produit peut être déduite du dossier qui contient son image).

**Résultats actuels** : seulement 3 images dans le catalogue de test. Ces essais valident le fonctionnement (une image du catalogue se retrouve en tête avec un score de 1, les fichiers invalides sont rejetés), mais **aucune conclusion statistique** n'est possible. La mesure de pertinence (Precision@K) sera faite avec le catalogue complet.

---

## 8. Limites connues

- **Ressemblance visuelle ne veut pas dire même fonction** : deux objets de catégories différentes mais d'apparence proche (couleur, forme) peuvent obtenir un score élevé.
- **Modèle généraliste** : ResNet50 est entraîné sur ImageNet (1000 classes), pas sur des produits e-commerce. Certaines catégories seront moins bien distinguées.
- **Qualité de la photo** : fond chargé, éclairage, angle ou objet partiellement visible dégradent la qualité du vecteur par rapport aux images propres du catalogue.
- **Redimensionnement simple** : l'image est étirée en 224 × 224 sans conserver ses proportions, ce qui déforme les objets allongés.
- **Interprétation des scores** : le score sert à comparer les produits entre eux ; un score de 0,9 ne signifie pas « 90 % de chances qu'il s'agisse du même produit ».
- **Recherche exhaustive** : la requête est comparée à tous les vecteurs. C'est adapté à un petit catalogue ; FAISS serait une amélioration pour un catalogue volumineux.

---

## 9. Sources et ressources externes

- Modèle : ResNet50 (He et al., *Deep Residual Learning for Image Recognition*), poids pré-entraînés sur ImageNet, fournis par Keras Applications.
- Bibliothèques : TensorFlow/Keras 2.21.0, NumPy 2.5.5, scikit-learn 1.9.1, Pillow 13.0.0.
- Images du catalogue et de test : à compléter avec la provenance de chaque image (site, auteur, licence).