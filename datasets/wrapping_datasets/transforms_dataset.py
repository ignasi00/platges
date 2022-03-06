
import albumentations as A
import torch
from torch.utils.data import Dataset


class TransformDataset(Dataset):

    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        data = self.dataset[idx]
        transformed = self.transforms(image=data[0], mask=data[1])
        image = transformed["image"]
        mask = transformed["mask"]

        return image, mask, *data[2:]

    def __len__(self):
        return len(self.dataset)
