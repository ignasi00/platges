
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
import cv2
import numpy as np
import os
import pandas as pd
import pickle
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

class Platges_ArgusNLDataset(Dataset):

    def _tabulate_files(self, folder_path, img_ext=None, seg_ext=None, cls_ext=None):
        img_ext = img_ext or '.jpg'
        seg_ext = seg_ext or '.segments.pkl' # TODO: maybe converting pkl to pt to speed up
        cls_ext = cls_ext or '.classes.pkl'

        folder_path = os.path.abspath(folder_path)

        list_of_files = pd.DataFrame({'filenames' : [f'{folder_path}/{f.name}' for f in os.scandir(folder_path) if f.is_file()]})

        list_of_items = pd.DataFrame()
        list_of_items[IMAGES] = list_of_files[list_of_files['filenames'].str.endswith(img_ext, na=False)]
        list_of_items[SEGMENTS] = list_of_items[IMAGES].str[:-len(img_ext)] + seg_ext
        list_of_items[CLASSES] = list_of_items[IMAGES].str[:-len(img_ext)] + cls_ext

        list_of_files = list_of_files['filenames']
        list_of_items = list_of_items[list_of_items[SEGMENTS].isin(list_of_files) & list_of_items[CLASSES].isin(list_of_files)]
        list_of_items.reset_index(drop=True)

        return list_of_items


    def __init__(self, folder_path, labels_map=None, aug=None, to_tensor=True, downsample=None, img_ext=None, seg_ext=None, cls_ext=None, default_value=-1):
        self.table_of_items = self._tabulate_files(folder_path, img_ext, seg_ext, cls_ext)
        self.labels_map = labels_map or LABELS
        self.default_value = default_value

        self.to_tensor = to_tensor

        aug = aug or []
        aug = aug + [ToTensor()] if to_tensor else aug
        self.aug = A.Compose(aug) if len(aug) > 0 else None

        self.downsample = downsample

    def __getitem__(self, idx):

        rows = self.table_of_items.iloc[idx]
        img_path, seg_path, cls_path = [rows[IMAGES], rows[SEGMENTS], rows[CLASSES]]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR) # TODO: what is the image shape?
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[8:-8, :, :] # there is a cropped version with image filename prefix cropped_

        if self.downsample is not None and self.downsample != 1:
            image = cv2.resize(image, (image.shape[0] // self.downsample, image.shape[1] // self.downsample), interpolation=cv2.INTER_AREA)

        if self.aug is not None:
            image = self.aug(image=image)['image']
            # image = np.transpose(image, (2, 0, 1)).astype(np.float32) or permute if tensor ¿?
        
        with open(seg_path, 'rb') as f:
            segments = pickle.load(f, encoding='latin1')
        with open(cls_path, 'rb') as f:
            classes = pickle.load(f)
        
        if labels_map is not None:
            classes = [labels_map.get(k, self.default_value) for k in classes]
        if self.to_tensor:
            segments = torch.tensor(segments, dtype=torch.long)
            classes = torch.tensor(classes, dtype=torch.long)

        return image, segments, classes, img_path

    def __len__(self):
        return len(self.table_of_items.index)
