import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generar datos de ejemplo
np.random.seed(0)
X = np.random.rand(100, 2)

# Crear un modelo K-Means con 3 clústeres
modelo_kmeans = KMeans(n_clusters=3)

# Ajustar el modelo a los datos
modelo_kmeans.fit(X)

# Obtener las etiquetas de clúster para cada punto de datos
etiquetas = modelo_kmeans.labels_

# Obtener las coordenadas de los centroides de clúster
centroides = modelo_kmeans.cluster_centers_

# Visualizar los datos y los clústeres
plt.scatter(X[:, 0], X[:, 1], c=etiquetas, cmap='viridis')
plt.scatter(centroides[:, 0], centroides[:, 1], marker='o', c='red', s=200)
plt.title("Agrupamiento de Datos con K-Means")
plt.show()
