import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, \
    BatchNormalization, Add, ReLU, ZeroPadding2D, GlobalAveragePooling2D, Input
from helper_functions import image_standardisation

# Code inspired by:
# https://github.com/AarohiSingla/ResNet50/blob/master/3-resnet50_rooms_dataset.ipynb

INPUT_SHAPE = (224, 224, 3)


def two_layer_identity_block(tensor, num_filters, stage):
    """
    Creates a two-layer identity block for a ResNet model.
    :param tensor: input tensor for the block
    :param num_filters: Int. Number of filters for the convolutional layers
    :param stage: Int. Counter for numbering the identity blocks
    :return: output tensor
    """

    # Saving the input tensor
    shortcut_tensor = tensor

    # 1st convolutional block
    tensor = Conv2D(filters=num_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    name=f"Conv1_{stage}")(tensor)  # Padding to keep dimensions

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm1_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU1_{stage}")(tensor)

    # 2nd convolutional block
    tensor = Conv2D(filters=num_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer="glorot_normal",
                    name=f"Conv2_{stage}")(tensor)  # Padding to keep dimensions

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm2_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU2_{stage}")(tensor)

    # Skip connection
    tensor = Add(name=f"SkipConnection_{stage}")([tensor, shortcut_tensor])
    tensor = ReLU(name=f"ReLU3_{stage}")(tensor)

    return tensor


def two_layer_projection_block(tensor, num_filters, stage):
    """
    Creates a two-layer projection block for a ResNet model.
    :param tensor: input tensor for the block
    :param num_filters: Int. Number of filters for the convolutional layers
    :param stage: Int. Counter for numbering the identity blocks
    :return: output tensor
    """

    # Saving the unaltered tensor to add to the convolutional output later
    shortcut_tensor = tensor

    # 1st convolutional layer
    tensor = Conv2D(filters=num_filters,
                    kernel_size=(3, 3),
                    strides=(2, 2),  # Strides (2, 2) to reduce the dimensions
                    padding="same",  # Padding to obtain even half of the dimensions
                    kernel_initializer="glorot_normal",
                    name=f"Conv1_{stage}")(tensor)

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm1_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU1_{stage}")(tensor)

    # 2nd convolutional layer
    tensor = Conv2D(filters=num_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer="glorot_normal",
                    name=f"Conv2_{stage}")(tensor)  # Padding to keep dimensions

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm2_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU2_{stage}")(tensor)

    # 1x1 convolution for the shortcut tensor
    shortcut_tensor = Conv2D(filters=num_filters,
                             kernel_size=(1, 1),
                             strides=(2, 2),
                             kernel_initializer="glorot_normal",
                             name=f"Conv3_{stage}")(shortcut_tensor)  # strides=(2,2) for dimension reduction

    shortcut_tensor = BatchNormalization(axis=3,
                                         name=f"BatchNorm3_{stage}")(shortcut_tensor)

    # Skip connection
    tensor = Add(name=f"SkipConnection_{stage}")([tensor, shortcut_tensor])
    tensor = ReLU(name=f"ReLU3_{stage}")(tensor)

    return tensor


def ResNet14(input_shape=INPUT_SHAPE):
    """
    Creates a 14 layer residual neural network
    :param input_shape: input shape of the tensors
    :return: ResNet14 model
    """

    # Input layer
    tensor_input = Input(shape=input_shape,
                         batch_size=32,
                         name="Input")

    # Preprocessing layers
    tensor = tf.keras.layers.RandomFlip(mode="horizontal",
                                        input_shape=INPUT_SHAPE,
                                        name="Random_horizontal_flip")(tensor_input)
    tensor = tf.keras.layers.RandomContrast(factor=0.3,
                                            name="Random_contrast")(tensor)
    tensor = tf.keras.layers.Lambda(function=image_standardisation,
                                    name="Per_image_standardisation")(tensor)

    # Convolutional Block
    tensor = ZeroPadding2D(padding=(3, 3),
                           name="ZeroPadding")(tensor)

    tensor = Conv2D(filters=64,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    kernel_initializer="glorot_normal",
                    name="Conv1")(tensor)

    tensor = BatchNormalization(axis=3,
                                name="BatchNorm1")(tensor)

    tensor = ReLU(name="ReLU1")(tensor)

    tensor = MaxPooling2D(pool_size=(3, 3),
                          strides=(2, 2),
                          padding="same",
                          name="MaxPool")(tensor)

    # 1st ResNet Block
    tensor = two_layer_identity_block(tensor,
                                      num_filters=64,
                                      stage=1)
    tensor = two_layer_identity_block(tensor,
                                      num_filters=64,
                                      stage=2)

    # 2nd ResNet Block
    tensor = two_layer_projection_block(tensor,
                                        num_filters=128,
                                        stage=3)

    tensor = two_layer_identity_block(tensor,
                                      num_filters=128,
                                      stage=4)

    # 3rd ResNet Block
    tensor = two_layer_projection_block(tensor,
                                        num_filters=256,
                                        stage=5)

    tensor = two_layer_identity_block(tensor,
                                      num_filters=256,
                                      stage=6)

    # Global Average Pooling
    tensor = GlobalAveragePooling2D(name="GlobalAvgPooling")(tensor)

    # Output layer
    tensor = Dense(units=1,
                   activation="sigmoid",
                   name="Output")(tensor)

    # Create model
    model = Model(inputs=tensor_input,
                  outputs=tensor,
                  name="ResNet14")

    return model


##########################

def three_layer_identity_block(tensor, num_filters, stage):
    """
    Creates a three-layer identity block for a ResNet model.
    :param tensor: input tensor for the block
    :param num_filters: Int. Number of filters for the convolutional layers
    :param stage: Int. Counter for numbering the identity blocks
    :return: output tensor
    """

    # Saving the unaltered tensor to add to the convolutional output later
    shortcut_tensor = tensor

    # 1st convolutional block
    tensor = Conv2D(filters=num_filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="valid",
                    kernel_initializer="glorot_normal",
                    name=f"Conv1_{stage}")(tensor)

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm1_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU1_{stage}")(tensor)

    # 2nd convolutional block
    tensor = Conv2D(filters=num_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer="glorot_normal",
                    name=f"Conv2_{stage}")(tensor)  # Padding to keep dimensions

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm2_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU2_{stage}")(tensor)

    # 3rd convolutional block
    tensor = Conv2D(filters=num_filters * 4,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="valid",
                    kernel_initializer="glorot_normal",
                    name=f"Conv3_{stage}")(tensor)

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm3_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU3_{stage}")(tensor)

    # Skip connection
    tensor = Add(name=f"SkipConnection_{stage}")([tensor, shortcut_tensor])
    tensor = ReLU(name=f"ReLU4_{stage}")(tensor)

    return tensor


def three_layer_projection_block(tensor, num_filters, stage, strides=(2, 2)):
    """
    Creates a three-layer projection block for a ResNet model.
    :param tensor: input tensor for the block
    :param num_filters: Int. Number of filters for the convolutional layers
    :param stage: Int. Counter for numbering the identity blocks
    :param strides: Tuple of the strides
    :return: output tensor
    """

    # Saving the unaltered tensor to add to the convolutional output later
    shortcut_tensor = tensor

    # 1st convolutional block
    tensor = Conv2D(filters=num_filters,
                    kernel_size=(1, 1),
                    strides=strides,  # Strides are (2, 2) to reduce the dimensions
                    kernel_initializer="glorot_normal",
                    padding="valid",
                    name=f"Conv1_{stage}")(tensor)

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm1_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU1_{stage}")(tensor)

    # 2nd convolutional block
    tensor = Conv2D(filters=num_filters,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    kernel_initializer="glorot_normal",
                    padding="same",  # Padding to keep dimensions
                    name=f"Conv2_{stage}")(tensor)

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm2_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU2_{stage}")(tensor)

    # 3rd convolutional block
    tensor = Conv2D(filters=num_filters * 4,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    kernel_initializer="glorot_normal",
                    padding="valid",
                    name=f"Conv3_{stage}")(tensor)

    tensor = BatchNormalization(axis=3,
                                name=f"BatchNorm3_{stage}")(tensor)

    tensor = ReLU(name=f"ReLU3_{stage}")(tensor)

    # 1x1 convolution for the shortcut tensor
    shortcut_tensor = Conv2D(filters=num_filters * 4,
                             kernel_size=(1, 1),
                             strides=strides,  # Strides are (2, 2) to reduce the dimensions
                             padding="valid",
                             kernel_initializer="glorot_normal",
                             name=f"Conv4_{stage}")(shortcut_tensor)

    shortcut_tensor = BatchNormalization(axis=3,
                                         name=f"BatchNorm4_{stage}")(shortcut_tensor)

    # Skip connection
    tensor = Add(name=f"SkipConnection_{stage}")([tensor, shortcut_tensor])
    tensor = ReLU(name=f"ReLU4_{stage}")(tensor)

    return tensor


def ResNet32(input_shape=INPUT_SHAPE):
    """
    Creates a 32 layer residual neural network
    :param input_shape: input shape of the tensors
    :return: ResNet32 model
    """

    # Define the input as a tensor with shape input_shape
    tensor_input = Input(shape=input_shape,
                         batch_size=32,
                         name="Input")

    # Preprocessing layers
    tensor = tf.keras.layers.RandomFlip(mode="horizontal",
                                        input_shape=INPUT_SHAPE,
                                        name="Random_horizontal_flip")(tensor_input)
    tensor = tf.keras.layers.RandomContrast(factor=0.3,
                                            name="Random_contrast")(tensor)
    tensor = tf.keras.layers.Lambda(function=image_standardisation,
                                    name="Per_image_standardisation")(tensor)

    # Convolutional Block
    tensor = ZeroPadding2D(padding=(3, 3),
                           name="ZeroPadding")(tensor)

    tensor = Conv2D(filters=64,
                    kernel_size=(7, 7),
                    strides=(2, 2),
                    kernel_initializer="glorot_normal",
                    name="Conv1")(tensor)

    tensor = BatchNormalization(axis=3,
                                name="BatchNorm1")(tensor)

    tensor = ReLU(name="ReLU1")(tensor)

    tensor = MaxPooling2D(pool_size=(3, 3),
                          strides=(2, 2),
                          padding="same",
                          name="MaxPool")(tensor)

    # 1st ResNet Block
    tensor = three_layer_projection_block(tensor,
                                          num_filters=64,
                                          stage=1,
                                          strides=(1, 1))

    tensor = three_layer_identity_block(tensor,
                                        num_filters=64,
                                        stage=2)

    tensor = three_layer_identity_block(tensor,
                                        num_filters=64,
                                        stage=3)

    # 2nd ResNet Block
    tensor = three_layer_projection_block(tensor,
                                          num_filters=128,
                                          stage=4)

    tensor = three_layer_identity_block(tensor,
                                        num_filters=128,
                                        stage=5)

    tensor = three_layer_identity_block(tensor,
                                        num_filters=128,
                                        stage=6)

    tensor = three_layer_identity_block(tensor,
                                        num_filters=128,
                                        stage=7)

    # 3rd ResNet Block
    tensor = three_layer_projection_block(tensor,
                                          num_filters=256,
                                          stage=8)

    tensor = three_layer_identity_block(tensor,
                                        num_filters=256,
                                        stage=9)

    tensor = three_layer_identity_block(tensor,
                                        num_filters=256,
                                        stage=10)

    # Global Average Pooling
    tensor = GlobalAveragePooling2D(name="GlobalAvgPooling")(tensor)

    # Output layer
    tensor = Dense(units=1,
                   activation="sigmoid",
                   name="Output")(tensor)

    # Create model
    model = Model(inputs=tensor_input,
                  outputs=tensor,
                  name="ResNet32")

    return model


# Creating the models
res_14 = ResNet14(input_shape=INPUT_SHAPE)
res_32 = ResNet32(input_shape=INPUT_SHAPE)

# Saving the models
res_14.save(
    "/content/drive/MyDrive/MachineLearningProject/models/resnet/res_14.keras",
    save_format="keras",
    overwrite=True)

res_32.save(
    "/content/drive/MyDrive/MachineLearningProject/models/resnet/res_32.keras",
    save_format="keras",
    overwrite=True)
