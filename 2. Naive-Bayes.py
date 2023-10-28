from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Conjunto de datos de entrenamiento ficticio
correos = [
    "oferta ganadora gratis",
    "ganador felicidades envío gratuito",
    "mejor precio oferta limitada compra ahora",
    "reunión mañana informe importante proyecto finalizado",
    "confirmación de la cita actualización del proyecto novedades de la empresa",
]

etiquetas = [1, 1, 1, 0, 0]  # 1 para "spam", 0 para "no spam"

# Crear un vectorizador de texto
vectorizador = CountVectorizer()
X = vectorizador.fit_transform(correos)

# Crear un clasificador Naïve-Bayes
clasificador = MultinomialNB()
clasificador.fit(X, etiquetas)

# Clasificar nuevos correos
nuevos_correos = ["ganador del concurso", "informe importante de ventas"]

# Transformar los nuevos correos en vectores
nuevos_correos_vectores = vectorizador.transform(nuevos_correos)

# Predecir las etiquetas
etiquetas_predichas = clasificador.predict(nuevos_correos_vectores)

for i, correo in enumerate(nuevos_correos):
    if etiquetas_predichas[i] == 1:
        print(f'El correo "{correo}" es SPAM.')
    else:
        print(f'El correo "{correo}" NO es SPAM.')
