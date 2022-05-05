
import torch


def build_segmentation_collate(device=None):
    device = device or torch.device('cpu')
    
    def collate_fn(batch):
            inputs = []
            targets = []

            for data in batch:
                input_, target = data[:2]
                inputs.append(torch.FloatTensor(input_))
                targets.append(torch.IntTensor(target))

            inputs = torch.stack(inputs)
            inputs = inputs.to(device)
            targets = torch.stack(targets)
            targets = targets.to(device)

            return inputs, targets
    
    return collate_fn


def build_segmentation_test_collate(device=None):
    device = device or torch.device('cpu')
    
    def collate_fn(batch):
            inputs = []
            targets = []
            img_pathes = []

            for input_, target, _, img_path in batch:
                inputs.append(torch.FloatTensor(input_))
                targets.append(torch.IntTensor(target))
                img_pathes.append(img_path)

            inputs = torch.stack(inputs)
            inputs = inputs.to(device)
            targets = torch.stack(targets)
            targets = targets.to(device)

            return inputs, targets, img_pathes
    
    return collate_fn

