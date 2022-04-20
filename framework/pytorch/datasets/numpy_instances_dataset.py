
import cv2
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset


# Constants only used internally:
IMAGES = 'images'
SEGMENTS = 'segments'
CLASSES = 'classes'


class NumpyInstancesDataset(Dataset):

    def __init__(self, list_path, data_root='', read_flag=cv2.IMREAD_COLOR):
        self.table_of_pathes = pd.read_csv(list_path, header=None, index_col=False, names=[IMAGES, SEGMENTS, CLASSES])
        self.read_flag = read_flag
        self.data_root = data_root
    
    def __getitem__(self, idx):

        rows = self.table_of_pathes.iloc[idx]
        img_path, seg_path, cls_path = [rows[IMAGES], rows[SEGMENTS], rows[CLASSES]]

        image = cv2.imread(f"{self.data_root}/{img_path}", self.read_flag) # (#row, #col, #color) (shape = H, W, 3)
        if self.read_flag == cv2.IMREAD_COLOR : image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(seg_path, 'rb') as f:
            segments = pickle.load(f, encoding='latin1') # ArgusNL: image mask of superpixels annoted with the index of the classes vector
        with open(cls_path, 'rb') as f:
            classes = pickle.load(f) # ArgusNL: vector that contains the categorical classes (with repetitions) of the previous superpixels (idx is idx_superpixel, value is class)
                                     # BCN: dict with {mask_value : class}
            
        return image, segments, classes, img_path

    def __len__(self):
        return len(self.table_of_pathes.index)
