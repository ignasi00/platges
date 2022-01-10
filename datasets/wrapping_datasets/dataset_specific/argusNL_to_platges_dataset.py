
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGES = 'images'
SEGMENTS = 'segments'
CLASSES = 'classes'

LABELS = {
    'sandbeach' : 1,
    'sky' : 2,
    'watersea' : 3,
    'objectdune' : 4,
    'objectbeach' : 5,
    'vegetation' : 6,
    'waterpool' : 7,
    'sanddune' : 8,
    'objectsea' : 9
}

LABELS_P = {
    'sandbeach' : 2,
    'sky' : 0,
    'watersea' : 1,
    'objectdune' : 2,
    'objectbeach' : 2,
    'vegetation' : 0,
    'waterpool' : 1,
    'sanddune' : 2,
    'objectsea' : 1
}

WATER_ID = 1    # 21 # Indexes from ADE20K that does not colide with ArgusNL and there is a good color selected.
SAND_ID = 2     # 46
OTHERS_ID = 0   #    # 0 does not overlap either.


def argus_to_platges(np_img, water_idxs=None, sand_idxs=None, water_id=WATER_ID, sand_id=SAND_ID, others_id=OTHERS_ID):
    water = water_idxs or [3, 7, 9]
    sand = sand_idxs or [1, 4, 5, 8]

    water_mask = np.zeros(np_img.shape)
    for i in water:
        water_mask = np.logical_or(water_mask, (np_img == i))
    
    sand_mask = np.zeros(np_img.shape)
    for i in sand:
        sand_mask = np.logical_or(sand_mask, (np_img == i))

    np_img[water_mask] = water_id
    np_img[sand_id] = sand_id
    np_img[(not water_mask) & (not sand_mask)] = others_id
    
    return np_img


class ArgusNL_to_PlatgesDataset():

    def __init__(self, argusNL_dataset, labels_map=None, default_value=-1):
        self.argusNL_dataset = argusNL_dataset
        self.labels_map = labels_map or LABELS_P
        self.default_value = default_value
    
    def __getitem__ (self, idx):
        image, segments, classes, img_path = self.argusNL_dataset[idx]
        classes = [self.labels_map.get(k, self.default_value) for k in classes]
        
        mask = segments.copy()
        for gt_idx, gt_cls in enumerate(classes):
            mask[segments == gt_idx] = gt_cls

        return image, mask, img_path

    def __len__(self):
        return len(self.argusNL_dataset)
