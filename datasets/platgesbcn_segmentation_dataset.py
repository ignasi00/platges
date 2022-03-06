
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .numpy_instances_dataset import NumpyInstancesDataset


# Recordatori, TODO: It is not sure water:1 and sand:2 or viseversa, classes is a dict of "number -> name (aigua, sorra)"
WATER_ID = 1
SAND_ID = 2
WATER_OR_SAND_ID = WATER_ID | SAND_ID # 1 | 2 == 3
OTHERS_ID = 0


class PlatgesBCNSegmentationDataset(Dataset):

    def __init__(self, list_path, data_root='', read_flag=cv2.IMREAD_COLOR):
        self.base_dataset = NumpyInstancesDataset(list_path, data_root, read_flag)
        
    def __getitem__(self, idx):
        image, mask, classes, img_path = self.base_dataset[idx]
        image = np.float32(image)

        return image, mask, classes, img_path
    
    def __len__(self):
        return len(self.base_dataset)