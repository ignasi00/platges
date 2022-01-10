
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


class ArgusNLDataset(Dataset):

    def __init__(self, list_path):
        self.table_of_items = pd.read_csv(list_path, header=None, index_col=False, names=[IMAGES, SEGMENTS, CLASSES])
    
    def __getitem__(self, idx):

        rows = self.table_of_items.iloc[idx]
        img_path, seg_path, cls_path = [rows[IMAGES], rows[SEGMENTS], rows[CLASSES]]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR) # TODO: what is the image shape? (#row, #col, #color) (shape = H, W, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[8:-8, :, :] # there is a cropped version with image filename prefix cropped_
        image = np.float32(image)

        with open(seg_path, 'rb') as f:
            segments = pickle.load(f, encoding='latin1') # ArgusNL: image mask of superpixels annoted with the index of the classes vector
        with open(cls_path, 'rb') as f:
            classes = pickle.load(f) #ArgusNL: vector that contains the categorical classes (with repetitions) of the previous superpixels
            
        return image, segments, classes, img_path

    def __len__(self):
        return len(self.table_of_items.index)
