import os
import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, image_folder, label_folder):

        self.image_folder = image_folder
        self.label_folder = label_folder

        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(label_folder))
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = np.load(img_path).astype(np.float32)

        if np.max(image) > np.min(image):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0,1]
            image = 2 * image - 1  # Normalize to [-1,1]
        
        else:
            image = np.zeros_like(image)

        image = torch.tensor(image, dtype = torch.float32)


        label_path = os.path.join(self.label_folder, self.label_files[idx])
        label = np.load(label_path).astype(np.float32)
        label = label.flatten()                                             #H4Pt is in 4x2

        # label = (label - np.min(label)) / (np.max(label) - np.min(label))  # Normalize to [0,1]

        label = (label+32)/(2*32)

        # label = label/32

        label = torch.tensor(label, dtype=torch.float32)

        return image, label

