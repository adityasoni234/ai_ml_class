import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

tf.random.set_seed(42)

iris = load_iris()
X = iris.data
y = iris.target

# Normalize the input features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print("X Train",X_train)
print("X Test",X_test)
print(y_train)
print(y_test)

model = keras.Sequential([
    keras.layers.Input(shape=(4,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Measures the difference between two probability distributions (actual and predicted).
model.fit(X_train,y_train,epochs=50, batch_size=1, verbose=2)

loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print("Accuracy :",accuracy)
