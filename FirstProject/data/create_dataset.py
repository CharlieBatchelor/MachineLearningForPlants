# Script to read in raw image files and create a uniform dataset that can be parsed to a training script.
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

images_path = "../data/images"
data = tf.keras.utils.image_dataset_from_directory(images_path, batch_size=5, image_size=[500, 500])
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next() # Grabs a batch of data - think this is kind of a list of batches?
# Images represented as numpy arrays
print("Batch shape: ", batch[0].shape)    # Images represented as numpy arrays, batch_size of them (5)
print("Batch labels: ", batch[1])
print("Image shape: ", batch[0][0].shape)

# From the batch, plot a few images that have been resized and added to dataset
fig, axs = plt.subplots(1, 4)
axs.ravel()
for i, im in enumerate(batch[0][:4]):
    axs[i].imshow(im.astype(int))
    axs[i].title.set_text(batch[1][i])
plt.show()
