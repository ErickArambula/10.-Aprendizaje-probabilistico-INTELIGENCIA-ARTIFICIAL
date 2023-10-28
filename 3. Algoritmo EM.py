import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(0)
datos = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
datos = datos.reshape(-1, 1)

# Crear un modelo de mezcla gaussiana con dos componentes
modelo_em = GaussianMixture(n_components=2)

# Ajustar el modelo a los datos (algoritmo EM)
modelo_em.fit(datos)

# Predecir las etiquetas de cluster para los datos
etiquetas = modelo_em.predict(datos)

# Obtener los parámetros estimados de los componentes
media_componentes = modelo_em.means_
varianza_componentes = modelo_em.covariances_

# Visualizar los datos y los clusters
plt.scatter(datos, np.zeros_like(datos), c=etiquetas, cmap='viridis')
plt.scatter(media_componentes, np.zeros_like(media_componentes), marker='o', c='red', s=200)
plt.title("Clustering de Datos con Algoritmo EM")
plt.show()
