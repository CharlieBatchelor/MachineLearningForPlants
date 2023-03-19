import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf

# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower]) # Is this how simple it is to create a data set?

print("Shape of images dataset is: ", images.shape)
batch_size, height, width, channels = images.shape

# Create two filters, this is manual creation. A CNN would learn them itself.
filters = np.zeros(shape=(7,7, channels, 2), dtype=np.float32) # x,y = 7,7 (size), channels = 3 (RGB), 2 (no. filters)
filters[:, 3, :, 0] = 1    # Vertical line
filters[3, :, :, 1] = 1    # Horizontal Line

print("Filter 1: \n", filters[:,:,0,0])
print("Filter 2: \n", filters[:,:,0,1])

# Convolve these filters over the input images
output = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs.ravel()
plot_images = [china, output[0, :, :, 0], output[0, :, :, 1]]
titles = ["Original Image", "Vertical Filter", "Horizontal Filter"]

for i, ax in enumerate(axs):
    ax.imshow(plot_images[i])
    ax.set_title(titles[i])
plt.show()

