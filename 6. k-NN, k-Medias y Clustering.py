import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Generar datos de ejemplo
np.random.seed(0)
X = np.concatenate([np.random.normal(0, 1, (100, 2)), np.random.normal(3, 1, (100, 2))])

# Visualizar los datos
plt.scatter(X[:, 0], X[:, 1], c='blue')
plt.title("Datos de Ejemplo")
plt.show()

# Aplicar k-Means para agrupar los datos en 2 clústeres
kmeans = KMeans(n_clusters=2)
etiquetas = kmeans.fit_predict(X)

# Visualizar los datos agrupados por k-Means
plt.scatter(X[:, 0], X[:, 1], c=etiquetas, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='o', c='red', s=200)
plt.title("Clustering con k-Means")
plt.show()

# Aplicar k-NN para encontrar los puntos más cercanos
knn = NearestNeighbors(n_neighbors=2)
knn.fit(X)
distancias, indices = knn.kneighbors(X)

# Visualizar los resultados de k-NN
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue')
plt.title("Datos de Ejemplo")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c='blue')
for i in range(len(X)):
    for j in indices[i, 1:]:
        plt.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1], 'red'])
plt.title("k-NN (k=2)")
plt.show()
