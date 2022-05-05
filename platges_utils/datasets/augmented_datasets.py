# if more data augmentation options are considered later, it may be tedious to adapt the project code...
# TODO: Make it more comfortable when working on a data augemntation project

import albumentations as A
import albumentations.pytorch
import cv2

from frameworks.pytorch.datasets.wrapping_datasets.transforms_dataset import TransformDataset

from .concat_dataset import build_concat_dataset


def build_train_dataset(dataset, resize_height, resize_width, crop_height, crop_width, mean, std, scale_limit, shift_limit, rotate_limit):

    transforms_list = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=scale_limit, shift_limit=shift_limit, rotate_limit=rotate_limit, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.Resize(resize_height, resize_width, interpolation=cv2.INTER_AREA, always_apply=True), 
        A.RandomCrop(crop_height, crop_width, p=1), #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        A.Normalize(mean=mean, std=std),
        A.pytorch.transforms.ToTensorV2()
    ]
    transforms = A.Compose(transforms_list)

    seg_dataset = TransformDataset(dataset, transforms, drop_extra_params=True)
    return seg_dataset

def build_val_dataset(dataset, resize_height, resize_width, crop_height, crop_width, mean, std):

    transforms_list = [
        A.Resize(resize_height, resize_width, interpolation=cv2.INTER_AREA, always_apply=True),
        A.RandomCrop(crop_height, crop_width, p=1),
        A.Normalize(mean=mean, std=std),
        A.pytorch.transforms.ToTensorV2()
    ]
    transforms = A.Compose(transforms_list)

    seg_dataset = TransformDataset(dataset, transforms, drop_extra_params=True)
    return seg_dataset

def build_test_dataset(dataset, resize_height, resize_width, mean, std):

    transforms_list = [
        A.Resize(resize_height, resize_width, interpolation=cv2.INTER_AREA, always_apply=True),
        A.Normalize(mean=mean, std=std),
        A.pytorch.transforms.ToTensorV2()
    ]
    transforms = A.Compose(transforms_list)

    seg_dataset = TransformDataset(dataset, transforms, drop_extra_params=False)
    return seg_dataset
