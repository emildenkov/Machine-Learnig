import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image = cv.imread('resources/digit.png')[:, :, 0]
image = np.invert(np.array([image]))

X_train = tf.keras.utils.normalize(x_train)
X_test = tf.keras.utils.normalize(x_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=3)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

prediction = model.predict(image)
print('Prediction: {}'.format(np.argmax(prediction)))
plt.imshow(image[0])
plt.show()
