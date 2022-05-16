
import json
import torch
from torch.utils.data import Dataset


class JSON_Dataset(Dataset):

    def __init__(self, list_path):
        with open(list_path) as f:
            self.list_of_pathes = json.load(f)

    def __getitem__(self, idx):
        rows = self.list_of_pathes[idx]
        return tuple(rows)
    
    def __len__(self):
        return len(self.list_of_pathes)
