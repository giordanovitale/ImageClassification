import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, ReLU
from helper_functions import image_standardisation

INPUT_SHAPE = (224, 224, 3)


# VGG model 1
vgg_1 = Sequential([
    # Preprocessing layers
    tf.keras.layers.RandomFlip(mode="horizontal",
                               input_shape=INPUT_SHAPE,
                               name="Random_horizontal_flip"),
    tf.keras.layers.RandomContrast(factor=0.3,
                                   name="Random_contrast"),
    tf.keras.layers.Lambda(function=image_standardisation,
                           name="Per_image_standardisation"),

    # 1st convolutional block
    Conv2D(filters=16,
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
    Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_3"),

    Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_4"),

    MaxPool2D(pool_size=(2, 2),
              strides=None,
              padding="valid",
              name="MaxPool_3"),

    Flatten(name="Flatten"),

    Dense(units=8,
          activation="relu",
          kernel_initializer="glorot_normal",
          bias_initializer="zeros",
          name="Dense"),

    Dense(units=1,
          activation="sigmoid",
          kernel_initializer="glorot_normal",
          bias_initializer="zeros",
          name="Output")
    ],
    name="VGG_1")


# VGG model 2
vgg_2 = Sequential([
    # Preprocessing layers
    tf.keras.layers.RandomFlip(mode="horizontal",
                               input_shape=INPUT_SHAPE,
                               name="Random_horizontal_flip"),
    tf.keras.layers.RandomContrast(factor=0.3,
                                   name="Random_contrast"),
    tf.keras.layers.Lambda(function=image_standardisation,
                           name="Per_image_standardisation"),

    # 1st convolutional block
    Conv2D(filters=16,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_1"),

    Conv2D(filters=16,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_2"),

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
           name="Conv_3"),

    Conv2D(filters=32,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_4"),

    MaxPool2D(pool_size=(2, 2),
              strides=None,
              padding="valid",
              name="MaxPool_2"),

    # 3rd convolutional block
    Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_5"),

    Conv2D(filters=64,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_6"),

    Conv2D(filters=64,
           kernel_size=(1, 1),
           strides=(1, 1),
           padding="valid",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_7"),

    MaxPool2D(pool_size=(2, 2),
              strides=None,
              padding="valid",
              name="MaxPool_3"),

    # 4th convolutional block
    Conv2D(filters=128,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_8"),

    Conv2D(filters=128,
           kernel_size=(3, 3),
           strides=(1, 1),
           padding="same",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_9"),

    Conv2D(filters=128,
           kernel_size=(1, 1),
           strides=(1, 1),
           padding="valid",
           activation="relu",
           kernel_initializer="glorot_normal",
           name="Conv_10"),

    MaxPool2D(pool_size=(2, 2),
              strides=None,
              padding="valid",
              name="MaxPool_4"),

    Flatten(name="Flatten"),

    Dense(units=16,
          activation="relu",
          kernel_initializer="glorot_normal",
          bias_initializer="zeros",
          name="Dense_1"),

    Dense(units=8,
          activation="relu",
          kernel_initializer="glorot_normal",
          bias_initializer="zeros",
          name="Dense_2"),

    Dense(units=1,
          activation="sigmoid",
          kernel_initializer="glorot_normal",
          bias_initializer="zeros",
          name="Output")
    ],
    name="VGG_2")


# Saving the models
vgg_1.save(
    "/content/drive/MyDrive/MachineLearningProject/models/vgg/vgg_1.keras",
    save_format="keras",
    overwrite=True)

vgg_2.save(
    "/content/drive/MyDrive/MachineLearningProject/models/vgg/vgg_2.keras",
    save_format="keras",
    overwrite=True)
