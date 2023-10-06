import os
from helper_functions import corrupted_images
from helper_functions import get_image_paths
from helper_functions import plot_images

# Set working directory
cwd = os.getcwd()

# Getting list of all images
image_paths = get_image_paths(directory=cwd)

# Checking for corrupted and non-jpg images
non_jpgs, corrupted = corrupted_images(images=image_paths)

# Showing random images
plot_images(image_paths)


