import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizar los datos y cambiar la forma de las imágenes
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Definir un modelo de red neuronal profunda
modelo = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = modelo.evaluate(test_images, test_labels)
print(f'Exactitud en el conjunto de prueba: {test_acc}')

# Visualizar una predicción
prediction = modelo.predict(test_images[0:1])
predicted_label = prediction.argmax()
print(f'Etiqueta predicha: {predicted_label}')

# Visualizar la imagen de prueba
plt.imshow(test_images[0].reshape(28, 28), cmap='viridis')
plt.title(f'Etiqueta verdadera: {test_labels[0]}, Etiqueta predicha: {predicted_label}')
plt.show()
