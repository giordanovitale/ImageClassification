import os
from math import sqrt
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf


def get_image_paths(directory=os.getcwd()):
    """
    Searches through a directory to find all subdirectories ending with 'muffin' or 'chihuahua'
    and appends all filepaths to a list.
    :param directory: Str. Directory to perform the search in
    :return: list of all filepaths found in the subdirectories
    """
    img_paths = list()

    for dirpath, dirnames, filenames in os.walk(directory):

        if dirpath.endswith("muffin") or dirpath.endswith("chihuahua"):
            print(f"Searching directory path: {dirpath}")

            for file in filenames:
                img_paths.append(f"{dirpath}/{file}")

    print(f"Number of images found: {len(img_paths)}")

    return img_paths


def corrupted_images(images):
    """
    Tries to open all files in a list of paths in binary mode and with the Image() function from PIL to find files
    that are not in jpg format or cannot be opened.
    :param images: List of image paths to try to open
    :return: list of non-jpg files and list of corrupted files
    """
    non_jpgs = list()
    corrupted = list()

    for img_path in images:

        if img_path.endswith("jpg"):
            try:
                image = open(img_path, mode="rb")
                image.read()
                image.close()
                Image.open(img_path)
            except Exception:
                corrupted.append(img_path)
                print(f"The file {img_path} cannot be opened")
        else:
            non_jpgs.append(img_path)

    print(f"Number of non-jpg files in the data: {len(non_jpgs)}")
    print(f"Number of corrupted files in the data: {len(corrupted)}")

    return non_jpgs, corrupted


def get_factors(num):
    """
    Calculates all integer factors of a given number.
    :param num: Number to calculate factors for
    :return: List of all integer factors
    """
    factors = list()

    if num > 1:
        for i in range(1, num + 1):
            if num % i == 0:
                factors.append(i)
    else:
        print("Insert a number higher than 1")

    return factors


def grid_shape(num):
    """
    Determines grid shape for plotting pictures.
    :param num: Number of pictures to be plotted
    :return: nrows: int. number of rows in the grid, ncols: int. number of columns in the grid
    """
    factors = get_factors(num)
    nrows, ncols = None, None

    if sqrt(num) in factors:
        nrows, ncols = sqrt(num), sqrt(num)
    else:
        for i in range(len(factors)):
            if sqrt(num) < factors[i]:
                nrows = factors[i - 1]
                ncols = factors[i]
                break

    return int(nrows), int(ncols)


def plot_random_images(image_paths, num=16, figsize=(10, 10)):
    """
    Plots a number of random sample images from the data set
    :param figsize: Tuple of the figure size
    :param image_paths: List image paths to plot images from
    :param num: Int. Number of images to plot
    :return: None
    """
    nrows, ncols = grid_shape(num)
    rand_imgs = np.reshape(random.sample(image_paths, num), (nrows, ncols))

    fig, ax = plt.subplots(figsize=figsize,  nrows=nrows, ncols=ncols)

    fig.suptitle("Sample of images from the data set")
    for i in range(ncols):
        for j in range(nrows):
            img = plt.imread(rand_imgs[i, j])
            ax[i, j].imshow(img)
            if "chihuahua" in rand_imgs[i, j]:
                title = "Chihuahua"
            else:
                title = "Muffin"
            ax[i, j].set_title(title)
            ax[i, j].axis("off")

    plt.show()


def history_to_json(history, file_path):
    """
    Saves the history of a tensorflow model to a .json file
    :param history: tensorflow model history
    :param file_path: Path of the .json file to write the history to
    :return: None
    """
    hist_dict = {"loss": history.history["loss"], "accuracy": history.history["accuracy"],
                 "val_loss": history.history["val_loss"], "val_accuracy": history.history["val_accuracy"]}
    json.dump(hist_dict, open(file_path, "w"))


def load_json(file_path):
    """
    Loads a .json file as a dictionary
    :param file_path: Path of the .json file to write the history to
    :return: Dict. of the .json file
    """
    history = json.load(open(file_path, "r"))
    return history


def plot_metrics(history, figsize=(10, 5)):
    """
    Plots the loss and accuracy curves from a history dictionary.
    :param history: Dict. of the history of a tensorflow model
    :param figsize: Tuple of the figure size
    :return: None
    """
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=2)

    loss = history["loss"]
    val_loss = history["val_loss"]

    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    epochs = range(1, 1 + len(history["loss"]))

    # Plot loss
    ax[0].plot(epochs, loss, label="Training Loss", linewidth=2, color="#0291b5")
    ax[0].plot(epochs, val_loss, label="Validation Loss", linewidth=2, color="#eb0076")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(epochs, accuracy, label="Training Accuracy", linewidth=2, color="#0291b5")
    ax[1].plot(epochs, val_accuracy, label="Validation Accuracy", linewidth=2, color="#eb0076")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].legend()

    plt.show()


def plot_augmented_image(image, augmentation_layer, num=9, figsize=(10, 10)):
    """
    Plots a number of random sample images from the data set
    :param image: Numpy array or tensor of image to augment
    :param augmentation_layer: Keras sequential stack of augmentation layers
    :param num: Int. Number of augmentations to plot
    :param figsize: Tuple of the figure size
    :return: None
    """

    nrows, ncols = grid_shape(num)

    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

    fig.suptitle("Image augmentation")

    for i in range(ncols):
        for j in range(nrows):
            aug_img = augmentation_layer(image)
            ax[i, j].imshow(aug_img)
            ax[i, j].axis("off")

    plt.show()


def random_invert_img(image, prob=0.2):
    """
    Randomly inverts the colours of an image with a given probability
    :param image: Numpy array or tensor of image to invert
    :param prob: Float Probability with which the image is inverted
    :return: image: Numpy array or tensor of the image
    """
    if random.uniform(a=0, b=1) < prob:
        x = (1 - image)
    else:
        image
    return image


def image_standardisation(tensor):
    """
    This is simply a workaround, as it is not possible to load models with lambda functions,
    unless you insert a custom function
    """
    return tf.image.per_image_standardization(tensor)
