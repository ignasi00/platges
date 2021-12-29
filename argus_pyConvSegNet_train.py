
import cv2
import numpy as np
import os
from PIL import Image, ImageMath
from skimage import io, color
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.platges_segmentation_dataset import Platges_ArgusNLDataset, LABELS
from metrics.mIoU import mIoU

from extern.pyconvsegnet.model.pyconvsegnet import PyConvSegNet
from extern.pyconvsegnet.tool.test import scale_process, colorize
from extern.pyConvSegNet_utils import apply_net_cpu


SEGMENTATION_PREFIX = 'seg_'
OVERLAPPED_PREFIX = 'ovr_'

RED = np.array([255, 0, 0])

WATER_ID = 21 # Indexes from ADE20K that does not colide with ArgusNL and there is a good color selected.
SAND_ID = 46
OTHERS_ID = 0 # 0 does not overlap either.


# TODO: improbe all label transformations from datasets to our problem (this method may crack in some other dataset case)
def argus_to_platges(np_img):
    water = [3, 7, 9]
    sand = [1, 4, 5, 8] # and 13
    for i in water:
        np_img[np_img == i] = WATER_ID
    for i in sand:
        np_img[np_img == i] = SAND_ID
    np_img[(np_img != WATER_ID) & (np_img != SAND_ID)] = OTHERS_ID
    return np_img

def ade20k_to_platges(np_img):
    water = [9, 21, 26, 37, 109, 113, 128]
    sand = [0, 46, 81, 94] # and 13
    for i in water:
        np_img[np_img == i] = WATER_ID
    for i in sand:
        np_img[np_img == i] = SAND_ID
    np_img[(np_img != WATER_ID) & (np_img != SAND_ID)] = OTHERS_ID
    return np_img

###############################################

def buid_dataloader(data_path, downsample=None):
    dataset = Platges_ArgusNLDataset(   data_path,
                                        labels_map=LABELS,
                                        to_tensor=True,
                                        downsample=downsample,
                                        img_ext=None,
                                        seg_ext=None,
                                        cls_ext=None,
                                        default_value=-1)

    def my_collate(x): return x # <- do not transform imgs to tensor here
    dataloader = DataLoader(dataset, collate_fn=my_collate)

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
    output = apply_net_cpu(segmentation_net, img, num_classes, crop_h, crop_w, mean, std, base_size, scales)

    if ade20k_labels : output = ade20k_to_platges(output)

    return output

def process_ground_truth(segments, classes=None, argus_labels=False):
    ground_truth = segments.clone()

    if classes is not None:
        for gt_idx, gt_cls in enumerate(classes):
            ground_truth[segments == gt_idx] = gt_cls
    
    if argus_labels : ground_truth = argus_to_platges(ground_truth)
    
    return ground_truth

def compute_metrics(seg_img, ground_truth, verbose=False):
    img_mIoU = mIoU(ground_truth, seg_img)
    if verbose : print(f'img_mIoU =  {img_mIoU}')
    return img_mIoU

def save_iter(img, seg_img, colors, save_path, img_path, alpha, error_red=False, ground_truth=None):
    save_path_seg = f"{save_path}/{SEGMENTATION_PREFIX}{os.path.basename(img_path)}"
    save_path_ovr = f"{save_path}/{OVERLAPPED_PREFIX}{os.path.basename(img_path)}"

    seg_img_col = colorize(seg_img, colors)
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

def main(data_path, downsample, argus_labels, layers, num_classes, ade20k_labels, zoom_factor, backbone_output_stride, backbone_net, pretrained_path, crop_h, crop_w, mean, std, base_size, scales, save_path, colors_path, alpha, error_red, verbose):
    os.makedirs(save_path, exist_ok = True)

    dataloader = buid_dataloader(data_path, downsample)

    segmentation_net = build_model(layers, num_classes, zoom_factor, backbone_output_stride, backbone_net, pretrained_path)

    colors = None
    if colors_path is not None: colors = np.loadtxt(colors_path).astype('uint8')

    v_mIoU = []
    for id_batch, batch in enumerate(dataloader):
        for img, segments, classes, img_path in batch:
            if verbose:
                print(f"[{id_batch} / {len(dataloader)}] - {img_path}")
            
            seg_img = process_data(segmentation_net, img, num_classes, crop_h, crop_w, mean, std, base_size, scales, ade20k_labels)
            ground_truth = process_ground_truth(segments, classes, argus_labels)
            v_mIoU.append(compute_metrics(seg_img, ground_truth, verbose))

            if colors is not None : save_iter(img, seg_img, colors, save_path, img_path, alpha, error_red, ground_truth)
    
    save_metrics(v_mIoU, save_path, verbose)


if __name__ == "__main__":

    VALUE_SCALE = 255

    data_path = '/mnt/c/Users/Ignasi/Downloads/ArgusNL'
    downsample = 4
    argus_labels = True
    layers = 152
    classes = 150
    ade20k_labels = True
    zoom_factor = 8
    backbone_output_stride = 8
    backbone_net = "pyconvresnet"
    pretrained_path = '/home/ignasi/platges/extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth'
    crop_h = 473
    crop_w = 473
    MEAN = [0.485, 0.456, 0.406]
    mean = [item * VALUE_SCALE for item in MEAN]
    STD = [0.229, 0.224, 0.225]
    std = [item * VALUE_SCALE for item in STD]
    base_size = 512
    scales = [1.0]
    save_path = '/mnt/c/Users/Ignasi/Downloads/argus_saved/'
    colors_path = "extern/pyconvsegnet/dataset/ade20k/ade20k_colors.txt"
    alpha = 0.5
    error_red = True
    verbose = True

    main(   data_path, downsample, argus_labels, layers, 
            classes, ade20k_labels, zoom_factor, backbone_output_stride, 
            backbone_net, pretrained_path, crop_h, crop_w,
            mean, std, base_size, scales, save_path, 
            colors_path, alpha, error_red, verbose)
