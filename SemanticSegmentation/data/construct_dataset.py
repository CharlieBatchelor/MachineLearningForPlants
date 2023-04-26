import os
from PIL import Image
from torch.utils.data import Dataset  # For testing
import numpy as np

class PlantDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  # List of files in that dir

    def __len__(self):
        return len(sel.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).conver("L"), dtype=np.float32)  # Convert to 0, 255, black and white!
        mask[mask == 255.0] = 1.0  # Using Sigmoid on final activation -> Probability of pixels!

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["images"]
            mask = augmentations["mask"]
            



