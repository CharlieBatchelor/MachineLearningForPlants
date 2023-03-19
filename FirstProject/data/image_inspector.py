# Script to inspect specific images from the dataset.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Process Arguments
parser = argparse.ArgumentParser(
    description="Accepts an input image and splits it up into it's constituent RGB images for plotting.")
parser.add_argument('-i', '--image', dest="image_location", default="../data/test_image/test_image.jpg")
args = parser.parse_args()

image = Image.open(args.image_location)
print("Format: ", image.format)
print("Image size: ", image.size)
print("Image mode: ", image.mode)

np_image = np.asarray(image)
print("\nnumpy image type: ", type(np_image))
print("numpy image shape: ", np_image.shape)
print("Number of layers: ", len(np_image[0, 0, :]))

# Do the plotting
cmaps = ["None", "Reds", "Greens", "Blues"]
titles = ["Original", "Reds", "Greens", "Blues"]
plot_images = [image, np_image[:, :, 0], np_image[:, :, 1],np_image[:, :, 2]]
num_layers = len(np_image[0, 0, :])
fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
fig.suptitle("Sample Image Decomposition", fontweight="bold")
for i, ax in enumerate(axs.ravel()):
    if i == 0:
        ax.imshow(plot_images[i])
    else:
        ax.imshow(plot_images[i], cmap=cmaps[i])
    ax.set_title(titles[i], fontweight="bold")
plt.show()

