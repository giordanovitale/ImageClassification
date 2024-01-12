import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, \
    AveragePooling2D, BatchNormalization, ReLU, PReLU
from helper_functions import image_standardisation


INPUT_SHAPE = (224, 224, 3)

# Sequential Model 1
seq_1 = Sequential([
    # Preprocessing layers
    tf.keras.layers.RandomFlip(mode="horizontal",
                               input_shape=INPUT_SHAPE,
                               name="Random_horizontal_flip"),
    tf.keras.layers.RandomContrast(factor=0.3,
                                   name="Random_contrast"),
    tf.keras.layers.Lambda(function=image_standardisation,
                           name="Per_image_standardisation"),

    # 1st convolutional block
    Conv2D(filters=32,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_1"),

    MaxPool2D(pool_size=(2, 2),
              strides=None,
              padding="valid",
              name="MaxPool_1"),

    # 2nd convolutional block
    Conv2D(filters=32,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_2"),

    MaxPool2D(pool_size=(2, 2),
              strides=None,
              padding="valid",
              name="MaxPool_2"),

    # 3rd convolutional block
    Conv2D(filters=32,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_3"),

    MaxPool2D(pool_size=(2, 2),
              strides=None,
              padding="valid",
              name="MaxPool_3"),

    Flatten(name="Flatten"),

    Dense(units=1,
          activation="sigmoid",
          kernel_initializer="glorot_normal",
          bias_initializer="zeros",
          name="Output")
],
    name="Sequential_1")


# Sequential model 2
seq_2 = Sequential([
    # Preprocessing layers
    tf.keras.layers.RandomFlip(mode="horizontal",
                               input_shape=INPUT_SHAPE,
                               name="Random_horizontal_flip"),
    tf.keras.layers.RandomContrast(factor=0.3,
                                   name="Random_contrast"),
    tf.keras.layers.Lambda(function=image_standardisation,
                           name="Per_image_standardisation"),

    # 1st convolutional block
    Conv2D(filters=128,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation=None,
           kernel_initializer="HeNormal",
           name="Conv_1"),
    ReLU(name="ReLU_1"),
    AveragePooling2D(pool_size=(2, 2),
                     strides=None,
                     padding="valid",
                     name="AvgPool_1"),
    BatchNormalization(name="BatchNorm_1"),

    # 2nd convolutional block
    Conv2D(filters=128,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation=None,
           kernel_initializer="HeNormal",
           name="Conv_2"),
    ReLU(name="ReLU_2"),
    AveragePooling2D(pool_size=(2, 2),
                     strides=None,
                     padding="valid",
                     name="AvgPool_2"),
    BatchNormalization(name="BatchNorm_2"),

    # 3rd convolutional block
    Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation=None,
           kernel_initializer="HeNormal",
           name="Conv_3"),
    ReLU(name="ReLU_3"),
    AveragePooling2D(pool_size=(2, 2),
                     strides=None,
                     padding="valid",
                     name="AvgPool_3"),
    BatchNormalization(name="BatchNorm_3"),

    # 4th convolutional block
    Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation=None,
           kernel_initializer="HeNormal",
           name="Conv_4"),
    PReLU(name="PReLU_1"),
    AveragePooling2D(pool_size=(2, 2),
                     strides=None,
                     padding="valid",
                     name="AvgPool_4"),
    BatchNormalization(name="BatchNorm_4"),

    # 5th convolutional block
    Conv2D(filters=32,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation=None,
           kernel_initializer="HeNormal",
           name="Conv_5"),
    PReLU(name="PReLU_2"),
    AveragePooling2D(pool_size=(2, 2),
                     strides=None,
                     padding="valid",
                     name="AvgPool_5"),
    BatchNormalization(name="BatchNorm_5"),

    Flatten(name="Flatten"),

    Dense(units=1,
          activation="sigmoid",
          kernel_initializer="HeNormal",
          bias_initializer="zeros",
          name="Output")
],
    name="Sequential_2")


# Saving the models
seq_1.save(
    "/content/drive/MyDrive/MachineLearningProject/models/sequential/seq_1.keras",
    save_format="keras",
    overwrite=True)

seq_2.save(
    "/content/drive/MyDrive/MachineLearningProject/models/sequential/seq_2.keras",
    save_format="keras",
    overwrite=True)
