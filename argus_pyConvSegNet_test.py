
import cv2
import numpy as np
import os
from PIL import Image, ImageMath
from skimage import io, color
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.argusNL_dataset import ArgusNLDataset
from datasets.wrapping_datasets.transforms_dataset import TransformDataset
from datasets.wrapping_datasets.dataset_specific.argusNL_to_platges_dataset import ArgusNL_to_PlatgesDataset
from loggers.wandb import WandB_logger
from metrics.mIoU import mIoU

from extern.pyconvsegnet.model.pyconvsegnet import PyConvSegNet
from extern.pyconvsegnet.tool.test import scale_process
from extern.pyConvSegNet_utils import apply_net_eval_cpu


SEGMENTATION_PREFIX = 'seg_'
OVERLAPPED_PREFIX = 'ovr_'

GRAY = np.array([120 120 120])
RED = np.array([255, 0, 0])
BLUE = np.array([0, 0, 255])
YELLOW = np.array([255, 255, 0])

WATER_ID = 21 # Indexes from ADE20K that does not colide with ArgusNL and there is a good color selected.
SAND_ID = 46
OTHERS_ID = 0 # 0 does not overlap either.


def ade20k_to_platges(np_img, water_idxs=None, sand_idxs=None, water_id=WATER_ID, sand_id=SAND_ID, others_id=OTHERS_ID):
    water = water_idxs or [9, 21, 26, 37, 109, 113, 128]
    sand = sand_idxs or [0, 46, 81, 94] # and 13

    water_mask = np.zeros(np_img.shape)
    for i in water:
        water_mask = np.logical_or(water_mask, (np_img == i))
    
    sand_mask = np.zeros(np_img.shape)
    for i in sand:
        sand_mask = np.logical_or(sand_mask, (np_img == i))

    np_img[water_mask] = water_id
    np_img[sand_id] = sand_id
    np_img[(not water_mask) & (not sand_mask)] = others_id
    
    return np_img

###############################################

def buid_dataloader(data_path, resize_height, resize_width, default_value=-1):
    argusNL = ArgusNLDataset(data_path)
    platges = ArgusNL_to_PlatgesDataset(argusNL, default_value=default_value)
    
    transforms_list = [
        A.Resize(resize_height, resize_width, interpolation=interpolation=cv2.INTER_AREA, always_apply=True), #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        A.ToTensorV2(),
    ]
    transforms = A.Compose(transforms_list)

    dataset = TransformDataset(platges, transforms)

    dataloader = DataLoader(dataset, batch_size=1)

    return dataloader

def build_model(layers, num_classes, zoom_factor, backbone_output_stride, backbone_net, pretrained_path=None):
    model = PyConvSegNet(layers=layers, classes=num_classes, zoom_factor=zoom_factor,
                                    pretrained=False, backbone_output_stride=backbone_output_stride,
                                    backbone_net=backbone_net)

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path)
        model.load_state_dict(checkpoint, strict=False)

    return model

def process_data(segmentation_net, img, num_classes, crop_h, crop_w, mean, std, base_size, scales, ade20k_labels=False):
    with torch.no_grad():
        output = apply_net_eval_cpu(segmentation_net, img, num_classes, crop_h, crop_w, mean, std, base_size, scales)

    if ade20k_labels : output = ade20k_to_platges(output)

    return output

def compute_metrics(seg_img, ground_truth, verbose=False):
    img_mIoU = mIoU(ground_truth, seg_img)
    if verbose : print(f'img_mIoU =  {img_mIoU}')
    return img_mIoU

def save_iter(img, seg_img, save_path, img_path, alpha, error_red=False, ground_truth=None):
    save_path_seg = f"{save_path}/{SEGMENTATION_PREFIX}{os.path.basename(img_path)}"
    save_path_ovr = f"{save_path}/{OVERLAPPED_PREFIX}{os.path.basename(img_path)}"

    palette = np.array([GRAY, BLUE, YELLOW])
    seg_img_col = Image.fromarray(seg_img.astype(np.uint8)).convert('P')
    seg_img_col.putpalette(palette)
    seg_img_col.convert('RGB').save(save_path_seg)
    
    mask = np.array(seg_img_col.convert('RGB'), np.float64)
    if error_red : mask[ np.array(seg_img) != np.array(ground_truth) ] = RED

    img_with_seg = cv2.addWeighted(np.transpose(img.numpy().astype(np.float64), (1, 2, 0)), 1-alpha, mask, alpha, 0.0)
    cv2.imwrite(save_path_ovr, img_with_seg)

def save_metrics(v_mIoU, save_path, verbose=False):
    save_path_IoU = f"{save_path}/argus_IoUs"
    v_mIoU = np.array(v_mIoU)

    mean_mIoU = np.mean(v_mIoU)
    min_mIoU = np.min(v_mIoU)
    max_mIoU = np.max(v_mIoU)

    if verbose:
        print(f'\n----\n\nmean_mIou: {mean_mIoU}')
        print(f'min_mIou: {min_mIoU}')
        print(f'max_mIou: {max_mIoU}')

    v_mIoU.dump(f'{save_path_IoU}.npy')
    np.savetxt(f'{save_path_IoU}.txt', v_mIoU, fmt='%.3f')

def main(data_path, resize_height, resize_width, layers, num_classes, ade20k_labels, zoom_factor, 
            backbone_output_stride, backbone_net, pretrained_path, crop_h, crop_w, mean, std,
            base_size, scales, save, save_path, alpha, error_red, verbose):
    os.makedirs(save_path, exist_ok = True)

    dataloader = buid_dataloader(data_path, resize_height, resize_width)

    segmentation_net = build_model(layers, num_classes, zoom_factor, backbone_output_stride, backbone_net, pretrained_path)

    v_mIoU = []
    for id_batch, batch in enumerate(dataloader):
        for img, segments, classes, img_path in batch:
            if verbose:
                print(f"[{id_batch} / {len(dataloader)}] - {img_path}")
            
            seg_img = process_data(segmentation_net, img, num_classes, crop_h, crop_w, mean, std, base_size, scales, ade20k_labels)
            v_mIoU.append(compute_metrics(seg_img, ground_truth, verbose))

            if save : save_iter(img, seg_img, save_path, img_path, alpha, error_red, ground_truth)
    
    save_metrics(v_mIoU, save_path, verbose)


if __name__ == "__main__":

    VALUE_SCALE = 255

    data_path = '/home/ignasi/platges/data_lists/argusNL_test.csv'
    resize_height = int(1024 / 2)
    resize_width = int(1392 / 2)
    layers = 152
    classes = 3 # 150
    ade20k_labels = False
    zoom_factor = 8
    backbone_output_stride = 8
    backbone_net = "pyconvresnet"
    # pretrained_path = '/home/ignasi/platges/extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth'
    pretrained_path = '/home/ignasi/platges/model_parameters/test_model.pth'
    crop_h = 473
    crop_w = 473
    MEAN = [0.485, 0.456, 0.406]
    mean = [item * VALUE_SCALE for item in MEAN]
    STD = [0.229, 0.224, 0.225]
    std = [item * VALUE_SCALE for item in STD]
    base_size = 512
    scales = [1.0]
    save = True
    save_path = '/mnt/c/Users/Ignasi/Downloads/argus_saved/'
    alpha = 0.5
    error_red = True
    verbose = True

    main(   data_path, resize_height, resize_width, layers, 
            classes, ade20k_labels, zoom_factor, backbone_output_stride, 
            backbone_net, pretrained_path, crop_h, crop_w,
            mean, std, base_size, scales, save,
            save_path, alpha, error_red, verbose)
