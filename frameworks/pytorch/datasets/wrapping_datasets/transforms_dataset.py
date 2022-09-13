
import albumentations as A
import torch
from torch.utils.data import Dataset


class TransformDataset(Dataset):
    # TODO: change name to a more specified one (AlbuminationTransformDataset or similar, also it only consider images and masks)

    def __init__(self, dataset, transforms, drop_extra_params=False):
        self.dataset = dataset
        self.transforms = transforms
        self.drop_extra_params = drop_extra_params

    def __getitem__(self, idx):
        data = self.dataset[idx]
        transformed = self.transforms(image=data[0], mask=data[1])
        image = transformed["image"]
        mask = transformed["mask"]

        if self.drop_extra_params : return image, mask
        return image, mask, *data[2:]

    def __len__(self):
        return len(self.dataset)
