import os
import numpy as np
import tensorflow as tf
from helper_functions import corrupted_images, get_image_paths, plot_random_images, history_to_json, load_json, \
    plot_metrics, plot_augmented_image, random_invert_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

print("")
# Set working directory
print("Setting working directory")
cwd = os.getcwd()
print("")

# Getting list of all images
print("Obtaining image file paths")
image_paths = get_image_paths(directory=cwd)
print("")


# Checking for corrupted and non-jpg images
print("Checking for corrupted and non-jpg images in the image files")
non_jpgs, corrupted = corrupted_images(images=image_paths)
print("")

# Showing random images
print("Plotting a grid of random images")
plot_random_images(image_paths)
print("")

# Loading data set
print("Loading in the dataset as tensor batches")
print("")
dir_path = "/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML Project/data/train"

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(dir_path,
                                                               label_mode="binary",
                                                               color_mode="rgb",
                                                               batch_size=32,
                                                               image_size=(128, 128),
                                                               shuffle=True,
                                                               seed=42,
                                                               subset="both",
                                                               validation_split=0.25)


# Printing shapes of the elements in the training data set
print("")
print("Information about the tensor batches:")
print("")
for image_batch, labels_batch in train_ds:
    print(f"The training data set contains {len(train_ds)} batches")
    print("")
    print(f"Shape of an image batch: {image_batch.shape}")
    print(f"Shape of a label batch: {labels_batch.shape}")
    print("")
    print("Tensor of an image in the batch:")
    print(image_batch[0])
    print("")
    print(f"Maximum and minimum values of the image: {np.min(image_batch[0]), np.max(image_batch[0])}")
    print("")
    break

print("Normalising the data")
normalisation_layer = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalisation_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalisation_layer(x), y))
image_batch, labels_batch = next(iter(train_ds))
first_image = image_batch[0]
print(f"New maximum and minimum values of an image: {np.min(first_image), np.max(first_image)}")
print("")


print("Data Augmentation")
# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.Lambda(lambda x: random_invert_img(x, 0.2))
])

aug_train = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
aug_val = val_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

plot_augmented_image(image=next(iter(train_ds))[0][0], augmentation_layer=data_augmentation)
print("")


# Model building
print("Defining the models")
print("")
model_1 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(128, 128, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation="sigmoid")
])

model_2 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(128, 128, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1, activation="sigmoid")
])

print("Compiling the models")
print("")
model_1.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_2.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

# Fitting the model
print("Fitting the models")
# Callbacks
early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=1)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 20))

history_1 = model_1.fit(train_ds,
                        epochs=3,
                        steps_per_epoch=len(train_ds),
                        validation_data=val_ds,
                        validation_steps=len(val_ds),
                        callbacks=[lr_scheduler, early_stopper])

history_2 = model_2.fit(aug_train,
                        epochs=3,
                        steps_per_epoch=len(aug_train),
                        validation_data=aug_val,
                        validation_steps=len(aug_val),
                        callbacks=[lr_scheduler, early_stopper])


print("Saving model history to json file")
print("")
# Saving the history to a json file

history_file_path_1 = "/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML " \
                      "Project/hists/history_1.json"
history_file_path_2 = "/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML " \
                      "Project/hists/history_1.json"

history_to_json(history=history_1, file_path=history_file_path_1)
history_to_json(history=history_2, file_path=history_file_path_2)

# Loading json file of the models history
print("Loading model history to json file")
print("")
# Loading json file of the models history
history_dict_1 = load_json(history_file_path_1)
history_dict_2 = load_json(history_file_path_2)

print(f"Dictionaries of the models history loaded from the json file: {history_dict_1}, {history_dict_2}")
print("")

# Plotting metrics curves
print("Plotting loss and accuracy curves of the model")
plot_metrics(history=history_dict_1)
plot_metrics(history=history_dict_2)
