
import cv2
from kornia.contrib import distance_transform
import numpy as np
import torch
from torch.utils.data import Dataset

from frameworks.pytorch.datasets.numpy_instances_dataset import NumpyInstancesDataset


# Recordatori, TODO: It is not sure water:1 and sand:2 or viseversa, classes is a dict of "number -> name (aigua, sorra)"
WATER_ID = 1
SAND_ID = 2
WATER_OR_SAND_ID = WATER_ID | SAND_ID # 1 | 2 == 3
OTHERS_ID = 0


class PlatgesBCNSegmentationDataset(Dataset):

    def __init__(self, list_path, data_root='', read_flag=cv2.IMREAD_COLOR):
        self.base_dataset = NumpyInstancesDataset(list_path, data_root, read_flag)
        
    def __getitem__(self, idx):
        image, mask, classes, img_path = self.base_dataset[idx]
        image = np.float32(image)
        mask = np.int32(mask)

        return image, mask, classes, img_path
    
    def __len__(self):
        return len(self.base_dataset)

def resolve_ambiguity_platgesBCN(model_output, target, resolve=True):
    # output & target are batches of masks (B x H x W)
    # if model_output is (B x C x H x W) => resolve=True apply argmax(C) (becomes B x H x W)
    original_target = target.detach().clone()

    output = model_output.max(1)[1] if resolve == True else model_output

    ambiguity_mask = (original_target == WATER_OR_SAND_ID)
    sand_mask = (output == SAND_ID)
    water_mask = (output == WATER_ID)
    other_mask = (output == OTHERS_ID)

    ambiguity_correct_mask = ambiguity_mask & (water_mask | sand_mask) # intersection of target WATER_OR_SAND_ID with output WATER_ID or SAND_ID
    ambiguity_incorrect_mask = ambiguity_mask & other_mask # intersection of target WATER_OR_SAND_ID with output OTHERS_ID (complementary)

    # If target is ambiguous and output is water or sand, the target becomes the output
    target[ambiguity_correct_mask] = output[ambiguity_correct_mask].type(target.dtype)

    # The rest of the ambiguous mask is the nearest class (water or sand); something like a watershed
    watershed_img = torch.zeros([target.shape[0], 2, *target.shape[1:]], dtype=torch.float32, device=model_output.device)
    watershed_img[:, 1, :, :] = (original_target == SAND_ID).type(torch.float32)
    watershed_img[:, 0, :, :] = (original_target == WATER_ID).type(torch.float32)

    watershed_img = distance_transform(watershed_img)
    watershed_img = watershed_img[:, 1, :, :] - watershed_img[:, 0, :, :]

    new_sand_mask = ambiguity_incorrect_mask & (watershed_img < 0)
    new_water_mask = ambiguity_incorrect_mask & (watershed_img >= 0)

    target[new_sand_mask] = SAND_ID
    target[new_water_mask] = WATER_ID

    return model_output, target
