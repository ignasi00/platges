
import torch


def build_concat_dataset(lists_path, dataset_type):
    dataset = []
    for list_path in lists_path:
        dataset.append(dataset_type(list_path))
    
    if len(dataset) == 1 : dataset = dataset[0]
    else : dataset = torch.utils.data.ConcatDataset(dataset)

    return dataset

