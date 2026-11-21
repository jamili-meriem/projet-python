import time
from cbir.cbir_engine import search_similar

debut = time.time()
resultats = search_similar("dataset/a1.jpg", top_k=10)
print("1re recherche :", round(time.time() - debut, 2), "s")
print(resultats)

debut = time.time()
search_similar("dataset/a2.jpg", top_k=10)
print("2e recherche  :", round(time.time() - debut, 2), "s")