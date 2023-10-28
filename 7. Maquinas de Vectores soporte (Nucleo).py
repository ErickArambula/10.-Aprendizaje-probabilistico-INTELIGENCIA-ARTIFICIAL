import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Generar datos de ejemplo (un conjunto de datos no lineal)
X, y = datasets.make_circles(n_samples=100, factor=0.5, noise=0.1)

# Crear un modelo SVM con núcleo (kernel) polinómico
modelo_svm = SVC(kernel='poly', degree=3, C=1.0)

# Ajustar el modelo a los datos
modelo_svm.fit(X, y)

# Predecir las etiquetas para puntos en una cuadrícula
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
Z = modelo_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualizar los datos y la frontera de decisión
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Clasificación con Máquina de Vectores Soporte (SVM)")
plt.show()
