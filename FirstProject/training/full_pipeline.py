# Script that runs a full machine learning pipeline. Dataset creation, labelling, model building, and training.
# The validation results are plotted in the output area.
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def create_datasets(data_location, image_size, bs):
    """
    Takes in the data location as a relative path. Will create a data set using the keras
    utils API which automatically resizes images to be the same size for feeding to a CNN.
    :param data_location: relative path to data.
    :return: training_data, validation_data, test_data
    """
    data = tf.keras.utils.image_dataset_from_directory(data_location, batch_size=bs, image_size=[image_size, image_size])
    data = data.map(lambda x, y: (x/255, y))
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()  # Grabs a batch of data - think this is kind of a list of batches?
    # Images represented as numpy arrays
    print("Batch shape: ", batch[0].shape, " Batch labels example: ", batch[1], "Image shape: ", batch[0][0].shape)
    # Split data into training and testing partitions
    print("Length of data: ", len(data))
    train_size = int(len(data)*.8)
    val_size = int(len(data)*.1)
    test_size = int(len(data)*.1)
    sum = train_size + val_size + test_size
    if sum < len(data):
        print("TRAIN SIZE: ", train_size, " VALIDATION SIZE: ", val_size, " TEST SIZE: ", test_size)
        print("You should change these to use up your data!")
    else:
        print("Got TRAIN SIZE: ", train_size, " VALIDATION SIZE: ", val_size, " TEST SIZE: ", test_size)
        print("Total sum: ", sum)
    training_data = data.take(train_size)
    validation_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size + val_size).take(test_size)
    return training_data, validation_data, test_data


def build_network(image_size):
    """
    Constructs a neural network and returns it, ready for training.
    :return:
    """
    model = Sequential()
    model.add(Conv2D(50, (6, 6), 3, activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(20, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(image_size, activation='relu'))
    model.add(Dense(17, activation='softmax'))
    model.summary()
    # model.compile()
    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    return model


def plot_performance_results(hist, plot_loc):
    """
    Plots the performance of the network throughout learning. Save these to the output
    area as well as the original logs.
    :param hist: The history object created from the fit() or training function.
    """
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("Performance Metrics - Throughout Learning", fontweight='bold')
    axs.ravel()
    plot_data = [[hist.history['loss'], hist.history['val_loss']], [hist.history['accuracy'], hist.history['val_accuracy']]]
    labels = ["Loss", "Validation Loss", "Accuracy", "Validation Accuracy"]
    axs[0].plot(hist.history['loss'], label=labels[0])
    axs[0].plot(hist.history['val_loss'], label=labels[1])
    axs[0].legend(loc='upper right')
    axs[1].plot(hist.history['accuracy'], label=labels[2])
    axs[1].plot(hist.history['val_accuracy'], label=labels[3])
    axs[1].legend(loc='upper left')
    axs[1].set_xlabel("Epoch - No. of full dataset passes", fontweight='bold')
    f = str(plot_loc + "/learning_performance.png")
    plt.savefig(f, format='png')


if __name__ == '__main__':
    # Process Arguments
    parser = argparse.ArgumentParser(
        description="Full CNN data science workflow, just provide a list of images in their label directories!")
    parser.add_argument('-d', '--data_location', dest="data_location", default="../data/images",
                        help="Relative path to data location. Should contain subdirectories corresponding to labels.")
    parser.add_argument('-l', '--logs_location', dest="logs_location", default="../output/logs",
                        help="Relative path to logs location. Will dump tensor flow logs here during training.")
    parser.add_argument('-p', '--plots_location', dest="plots_location", default="../output/plots",
                        help="Relative path to plots location. Will dump final performance plots here.")
    args = parser.parse_args()

    # configuration
    image_size = 800
    batch_size = 10
    num_epochs = 5

    # Create a Dataset
    training_data, validation_data, test_data = create_datasets(args.data_location, image_size, batch_size)
    # Build a model to train
    model = build_network(image_size)
    # Setup logging area and train the thing.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logs_location)
    history = model.fit(training_data, epochs=num_epochs, validation_data=validation_data, callbacks=[tensorboard_callback])
    plot_performance_results(history, args.plots_location)


