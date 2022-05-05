import torch
from torch.utils.data import Dataset


class FunnelCollateDataset(Dataset):

    def __init__(self, dataset, num_params, offset=0):
        self.dataset = dataset
        self.num_params = num_params
        self.offset = offset

    def __getitem__(self, idx):
        data = self.dataset[self.offset : (self.offset + self.num_params)]
        if len(data) < self.num_params : data.extend([None] * (self.num_params - len(data)))
        
        return data[0], *data[1:]

    def __len__(self):
        return len(self.dataset)
