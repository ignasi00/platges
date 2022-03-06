
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .numpy_instances_dataset import NumpyInstancesDataset
from .wrapping_datasets.label_mapping_dataset import LabelListMappingDataset


WATER_ID = 1    # 21 # Indexes from ADE20K that does not colide with ArgusNL and there is a good color selected.
SAND_ID = 2     # 46
OTHERS_ID = 0   #    # 0 does not overlap either.

LABELS_P = {
    'sandbeach' : SAND_ID,
    'sky' : OTHERS_ID,
    'watersea' : WATER_ID,
    'objectdune' : SAND_ID,
    'objectbeach' : SAND_ID,
    'vegetation' : OTHERS_ID,
    'waterpool' : WATER_ID,
    'sanddune' : SAND_ID,
    'objectsea' : WATER_ID
}

class ArgusNLDataset(Dataset):

    def __init__(self, list_path, data_root='', read_flag=cv2.IMREAD_COLOR, labels_map=None, default_value=-1):
        labels_map = labels_map or LABELS_P

        first_dataset = NumpyInstancesDataset(list_path, data_root, read_flag)
        self.base_dataset = LabelListMappingDataset(first_dataset, labels_map, modified_indexs=[2], default_value=default_value)

    def __getitem__(self, idx):
        image, segments, classes, img_path = self.base_dataset[idx]
        image = image[8:-8, :, :] # there is a cropped version with image filename prefix cropped_
        image = np.float32(image)

        mask = segments.copy()
        for gt_idx, gt_cls in enumerate(classes):
            mask[segments == gt_idx] = gt_cls

        return image, mask, classes, img_path
    
    def __len__(self):
        return len(self.base_dataset)
