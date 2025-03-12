import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

path = "/Users/adityasoni234/Downloads/Multi-class Weather Dataset"

# read image using tensorflow

train_ds = tf.keras.utils.image_dataset_from_directory(
path,
validation_split=0.2,
subset="training",
seed=123,
image_size=(100, 100),
batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
path,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(100, 100),
batch_size=32)

print(train_ds)

class_names = train_ds.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# images, labels = next(iter(train_ds))
# for i in range(min(9, len(images))):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# plt.show()

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 4

model = tf.keras.Sequential([
tf.keras.layers.Rescaling(4./255),
tf.keras.layers.Conv2D(32, 3, activation='relu'),
tf.keras.layers.MaxPooling2D(),
tf.keras.layers.Conv2D(32, 3, activation='relu'),
tf.keras.layers.MaxPooling2D(),
tf.keras.layers.Conv2D(32, 3, activation='relu'),
tf.keras.layers.MaxPooling2D(),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(num_classes)
])

model.compile(
optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

model.fit(
train_ds,
validation_data=val_ds,
epochs=5
)

# plot the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()

# remaining

# take use input from user
image_path = "/Users/adityasoni234/Downloads/images-4.jpeg"

image = tf.keras.utils.load_img(image_path, target_size=(100, 100))  # Resize image
input_arr = tf.keras.utils.img_to_array(image)  # Convert to array
input_arr = tf.expand_dims(input_arr, 0)  # Add batch dimension
input_arr /= 255.0  # Normalize

# Make a prediction
predictions = model.predict(input_arr)
score = tf.nn.softmax(predictions[0])  # Convert logits to probabilities

# Get the predicted class
predicted_class = class_names[np.argmax(score)]
print(f"Predicted class: {predicted_class}")
