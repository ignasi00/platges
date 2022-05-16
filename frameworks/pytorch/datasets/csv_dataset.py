
import pandas as pd
import torch
from torch.utils.data import Dataset


class CSV_Dataset(Dataset):

    def __init__(self, list_path, data_root='', names=None):
        self.table_of_pathes = pd.read_csv(list_path, header=None, index_col=False, names=names)

    def __getitem__(self, idx):
        rows = self.table_of_pathes.iloc[idx].to_list()
        return tuple(rows)
    
    def __len__(self):
        return len(self.table_of_pathes.index)
