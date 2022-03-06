
import cv2
import json
import torch
from torch.utils.data import Dataset


class ImagesGroupDataset(Dataset):

    def __init__(self, json_path, data_root='', read_flag=0, downsample=None):
        with open(json_path) as f:
            self.list_of_lists = json.load(f)
        self.read_flag = read_flag
        self.data_root = data_root
        self.downsample = downsample
    
    def __getitem__(self, idx):
        images = []
        pathes = []
        for path in self.list_of_lists[idx]:
            images.append(cv2.imread(f"{self.data_root}/{path}", self.read_flag))
            
            if self.read_flag == cv2.IMREAD_COLOR:
                images[-1] = cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB)

            if isinstance(self.downsample, int) and self.downsample > 0:
                images[-1] = cv2.resize(images[-1], None, fx=1/self.downsample, fy=1/self.downsample, interpolation=cv2.INTER_NEAREST)
            
            pathes.append(path)

        return zip(images, pathes)

    def __len__(self):
        return len(self.list_of_lists)
