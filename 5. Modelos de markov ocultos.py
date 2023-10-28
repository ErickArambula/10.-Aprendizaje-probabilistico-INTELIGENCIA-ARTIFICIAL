import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Definir el modelo HMM
modelo_hmm = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Generar datos de ejemplo
np.random.seed(0)
muestras, etiquetas = modelo_hmm.sample(100)

# Ajustar el modelo HMM a los datos observados
modelo_hmm.fit(muestras)

# Generar nuevas muestras utilizando el modelo HMM
muestras_generadas, _ = modelo_hmm.sample(100)

# Visualizar los datos observados y los datos generados
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(muestras[:, 0], label="Datos Observados")
plt.legend()
plt.title("Datos Observados")

plt.subplot(2, 1, 2)
plt.plot(muestras_generadas[:, 0], label="Datos Generados")
plt.legend()
plt.title("Datos Generados")

plt.tight_layout()
plt.show()
