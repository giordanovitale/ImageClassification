import os
from math import sqrt
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt


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

    if sqrt(num) in factors:
        nrows, ncols = sqrt(num), sqrt(num)
    else:
        for i in range(len(factors)):
            if sqrt(num) < factors[i]:
                nrows = factors[i - 1]
                ncols = factors[i]
                break

    return int(nrows), int(ncols)


def plot_images(image_paths, num=16):
    """
    Plots a number of random sample images from the data set
    :param image_paths: List image paths to plot images from
    :param num: Int. Number of images to plot
    :return: None
    """
    nrows, ncols = grid_shape(num)
    rand_imgs = np.reshape(random.sample(image_paths, num), (nrows, ncols))

    fig, ax = plt.subplots(figsize=(10, 10),  nrows=nrows, ncols=ncols)

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
