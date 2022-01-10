
import cv2
import json
import torch
from torch.utils.data import Dataset


class ImageGroupDataset(Dataset):

    def __init__(self, json_path, read_flag=0):
        with open(json_path) as f:
            self.list_of_lists = json.load(f)
        self.read_flag = read_flag
    
    def __getitem__(self, idx):
        images = []
        pathes = []
        for path in self.list_of_lists[idx]:
            images.append(cv2.imread(path, self.read_flag))
            pathes.append(path)

        return zip(images, pathes)

    def __len__(self):
        return len(self.list_of_lists)
