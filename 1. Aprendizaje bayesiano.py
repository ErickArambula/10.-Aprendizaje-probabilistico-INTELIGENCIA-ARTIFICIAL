# Conjunto de datos de entrenamiento ficticio
# Cada fila representa un correo y sus palabras
# La última columna indica si es "spam" (1) o "no spam" (0)
training_data = [
    ["oferta", "ganador", "gratis", 1],
    ["ganador", "felicidades", "envío gratuito", 1],
    ["mejor precio", "oferta limitada", "compra ahora", 1],
    ["reunión mañana", "informe importante", "proyecto finalizado", 0],
    ["confirmación de la cita", "actualización del proyecto", "novedades de la empresa", 0],
]

# Calcular la probabilidad a priori
total_correos = len(training_data)
total_spam = sum(1 for correo in training_data if correo[-1] == 1)
probabilidad_spam = total_spam / total_correos
probabilidad_no_spam = 1 - probabilidad_spam

# Calcular la probabilidad de palabras en "spam" y "no spam"
palabras_spam = {}
palabras_no_spam = {}

for correo in training_data:
    for palabra in correo[:-1]:
        if correo[-1] == 1:
            palabras_spam[palabra] = palabras_spam.get(palabra, 0) + 1
        else:
            palabras_no_spam[palabra] = palabras_no_spam.get(palabra, 0) + 1

# Función para clasificar un nuevo correo como "spam" o "no spam"
def clasificar_correo(nuevo_correo):
    probabilidad_spam_dado_correo = probabilidad_spam
    probabilidad_no_spam_dado_correo = probabilidad_no_spam

    for palabra in nuevo_correo:
        if palabra in palabras_spam:
            probabilidad_spam_dado_correo *= palabras_spam[palabra] / total_spam
        if palabra in palabras_no_spam:
            probabilidad_no_spam_dado_correo *= palabras_no_spam[palabra] / (total_correos - total_spam)

    return probabilidad_spam_dado_correo > probabilidad_no_spam_dado_correo

# Ejemplo de clasificación
nuevo_correo = ["oferta", "ganador", "gratis"]
es_spam = clasificar_correo(nuevo_correo)

if es_spam:
    print("El nuevo correo es SPAM.")
else:
    print("El nuevo correo NO es SPAM.")
