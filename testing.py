import os
import numpy as np
import tensorflow as tf
from helper_functions import corrupted_images, get_image_paths, plot_random_images, history_to_json, load_json, \
    plot_metrics, plot_augmented_image, random_invert_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

dir_path = "/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML Project/data/train"

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(dir_path,
                                                               label_mode="binary",
                                                               color_mode="rgb",
                                                               batch_size=32,
                                                               image_size=(64, 64),
                                                               shuffle=True,
                                                               subset="both",
                                                               validation_split=0.25)

print(len(train_ds))
train_ds = train_ds.take(10)
print(len(train_ds))

normalisation_layer = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalisation_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalisation_layer(x), y))

model_1 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(64, 64, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation="sigmoid")
])

model_1.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

history_1 = model_1.fit(train_ds,
                        epochs=10,
                        steps_per_epoch=len(train_ds),
                        validation_data=val_ds,
                        validation_steps=len(val_ds))

model_1.save("/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML Project/models/model_1")


test_ds = tf.keras.utils.image_dataset_from_directory("/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML Project/data/test",
                                                      label_mode="binary",
                                                      color_mode="rgb",
                                                      batch_size=32,
                                                      image_size=(64, 64),
                                                      shuffle=True,
                                                      seed=42)

model_loaded = tf.keras.models.load_model("/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML Project/models/model_1")

model_loaded.evaluate(test_ds)
