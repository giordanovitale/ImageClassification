import os
import numpy as np
import tensorflow as tf
from helper_functions import corrupted_images, get_image_paths, plot_random_images, history_to_json, load_json, \
    plot_metrics, plot_augmented_image, random_invert_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

dir_path = "/Users/giord/Documents/ImageClassification/data/train"

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(dir_path,
                                                               label_mode="binary",
                                                               color_mode="rgb",
                                                               batch_size=32,
                                                               image_size=(64, 64),
                                                               shuffle=True,
                                                               subset="both",
                                                               validation_split=0.25,
                                                               seed=42)

print(len(train_ds))
train_ds = train_ds.take(10)
print(len(train_ds))

normalisation_layer = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalisation_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalisation_layer(x), y))

# model_1 = Sequential([
#     Conv2D(10, 3, activation="relu", input_shape=(64, 64, 3)),
#     MaxPool2D(pool_size=(2, 2)),
#     Conv2D(10, 3, activation="relu"),
#     MaxPool2D(pool_size=(2, 2)),
#     Conv2D(10, 3, activation="relu"),
#     MaxPool2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(1, activation="sigmoid")
# ])
#
#     model_1.compile(loss="binary_crossentropy",
#                     optimizer=Adam(),
#                     metrics=["accuracy"])
#
# history_1 = model_1.fit(train_ds,
#                         epochs=10,
#                         steps_per_epoch=len(train_ds),
#                         validation_data=val_ds,
#                         validation_steps=len(val_ds))
#
# model_1.save("/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML Project/models/model_1")


test_ds = tf.keras.utils.image_dataset_from_directory("/Users/giord/Documents/ImageClassification/data/test",
                                                      label_mode="binary",
                                                      color_mode="rgb",
                                                      batch_size=32,
                                                      image_size=(64, 64),
                                                      shuffle=True,
                                                      seed=42)

# model_loaded = tf.keras.models.load_model("/Users/philip/Documents/Milano University/2. Semester/Machine Learning/ML Project/models/model_1")
#
# model_loaded.evaluate(test_ds)



def sequential_family(num_conv_layers,
                      input_shape,
                      num_filters,
                      kernel_size,
                      pool_size,
                      num_dense_layers,
                      num_units_dense,
                      train_data,
                      num_epochs,
                      val_data,
                      model_path,
                      model_name
                      ):
    """
    Creates, compiles, trains and saves sequential models given some customizable inputs
    :param num_conv_layers: number of convolutional layers we want to add
    :param input_shape: input shape of the images (e.g. (64,64,3))
    :param num_filters: number of filters for convolutional layers
    :param kernel_size: kernel size of convolutional layers (e.g. 3)
    :param pool_size: pool size for max-pooling layers (e.g. (2,2))
    :param num_dense_layers: number of dense layers we want to add
    :param num_units_dense: Number of units in each dense layer
    :param train_data: training data set used to fit the model
    :param num_epochs: number of epochs in the training procedure
    :param val_data: validation data set used to evaluate the fitting procedure
    :param model_path: where we want to store the model properties, parameters, ect
    :param model_name: name of the model
    :return
    """

    # Start the model
    model = Sequential()

    # Add as many convolutional layers as we want
    for _ in range(num_conv_layers):
        model.add(Conv2D(num_filters, kernel_size, activation = 'relu', input_shape = input_shape))
        model.add(MaxPool2D(pool_size=pool_size))

    # Add the dense layer, again as many as we want
    model.add(Flatten())
    for _ in range(num_dense_layers):
        model.add(Dense(num_units_dense, activation = 'relu'))

    # Output layer
    model.add(Dense(1, activation = 'relu'))

    # Compile the model
    model.compile(loss = 'binary_crossentropy',
                  optimizer = Adam(),
                  metrics = ['accuracy'])

    # Fit the model
    history = model.fit(train_data,
                        epochs = num_epochs,
                        steps_per_epoch = len(train_data),
                        validation_data = val_data,
                        validation_steps = len(val_data),
                        verbose = 2
                        )

    # Save the model
    model.save(model_path, model_name)

    return model, history


model1, history1 = sequential_family(num_conv_layers=3,
                                     input_shape=(64,64,3),
                                     num_filters=10,
                                     kernel_size=3,
                                     pool_size=(2,2),
                                     num_dense_layers=1,
                                     num_units_dense=32,
                                     train_data=train_ds,
                                     num_epochs=10,
                                     val_data=val_ds,
                                     model_path="/Users/giord/Documents/ImageClassification/models",
                                     model_name = 'model1'
                                     )

model1.summary()