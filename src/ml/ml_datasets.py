import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class OCTDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=False):
        # OCT Dataset class for loading images and masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))
        self.masks_filenames = sorted(f for f in os.listdir(mask_dir) if f.endswith('.png'))

        if len(self.image_filenames) != len(self.masks_filenames):
            raise ValueError(f'Number of images ({len(self.image_filenames)}) '
                            f'does not match number of masks ({len(self.masks_filenames)})')
    #

    def __len__(self):
        return len(self.image_filenames) # how many images (img-mask-pairs) are in the dataset
    #

    def crop(self, img: np.ndarray, mask: np.ndarray):
        # Crop image and mask to ensure divisibility by 32
        h, w = img.shape

        # height adjustments
        if h % 32 != 0:
            h_mod = h % 32
            h -= h_mod
            img, mask = img[:h, :], mask[:h, :]

        # width adjustments
        if w % 32 != 0:
            w_mod = w % 32
            w -= w_mod
            img, mask = img[:, :w], mask[:, :w]

        return img, mask
    #

    def __getitem__(self, i):
        image_path = os.path.join(self.image_dir, self.image_filenames[i])
        mask_path = os.path.join(self.mask_dir, self.masks_filenames[i])

        image = Image.open(image_path).convert('L') # greyscale
        image = np.array(image).astype('float32')
        image -= image.min()
        image /= image.max() # Normalize to [0,1]
        
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask).astype('int64')

        image, mask = self.crop(image, mask) # crops to ensure divisibility by 32

        if self.transform:
            augmented = self.transform(image=image, mask=mask) # type: ignore
            image = augmented['image']
            mask = augmented['mask']

        image = torch.from_numpy(image).unsqueeze(0) # Add channel dimension (1, height, width)
        mask = torch.from_numpy(mask).long() # expected from PyTorch
        
        return image, mask
    #
##
