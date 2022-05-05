
import numpy as np
import torch
from torch.utils.data import Dataset


class LabelListMappingDataset(Dataset):

    def __init__(self, base_dataset, labels_map, modified_indexs=None, default_value=-1):
        self.base_dataset = base_dataset
        self.labels_map = labels_map
        self.indexs = modified_indexs or [0]
        try:
            for _ in self.indexs : break
        except:
            self.indexs = [self.indexs]
        self.default_value = default_value

    def __getitem__(self, idx):
        try:
            base_output = list(self.base_dataset[idx])
        except:
            base_output = [self.base_dataset[idx]]

        for _idx in self.indexs:
            base_output[_idx] = [self.labels_map.get(k, self.default_value) for k in base_output[_idx]]
        
        return [base_output[0], *base_output[1:]]
    
    def __len__(self):
        return len(self.base_dataset)
