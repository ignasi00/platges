
import cv2
import json
import torch
from torch.utils.data import Dataset


class ImagesDataset(Dataset):

    def __init__(self, json_path, data_root='', read_flag=0):
        with open(json_path) as f:
            self.list_of_images = json.load(f)
        self.read_flag = read_flag
        self.data_root = data_root
    
    def __getitem__(self, idx):
        image = cv2.imread(f"{self.data_root}/{self.list_of_images[idx]}", self.read_flag)
        if self.read_flag == cv2.IMREAD_COLOR : image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        path = self.list_of_images[idx]

        return (image, path)

    def __len__(self):
        return len(self.list_of_images)
