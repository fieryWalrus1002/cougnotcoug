import os
import PIL
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import pandas as pd
from torchvision.io import read_image


class CougData(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # index_in_full_set = self.data_indices[idx]
        # return self.celsius_values[index_in_full_set], self.fahrenheit_values[index_in_full_set]
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
