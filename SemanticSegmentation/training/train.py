import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# Load of useful things for inspection later
# from utils import (
#     load_checkpoint
# )

# HYPER PARAMETERS
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMAGE_DIR = "../data/..."
TRAIN_MASK_DIR = "../data/..."
VAL_IMAGE_DIR = "../data/..."
VAL_MASK_DIR = "../data/..."


def train(loader, model, optimiser, loss_fn, scaler):
    """
    Performs the training of a given model.
    :param loader:
    :param model: The model we would like to train.
    :param optimiser: The optimisation function, "adam" for example.
    :param loss_fn: The cost of getting something wrong.
    :param scaler:
    :return:
    """
    loop = tqdm(loader)   # What is TDQM?

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward Pass
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward Pass
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step()
        scaler.update()

        # Update the TQDM Loop
        loop.set_postfix(loss=loss.item())

def main():

    # Define a dataset augmentation pipeline - using albumentations, open source.
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


if __name__ == "__main__":
    main()
