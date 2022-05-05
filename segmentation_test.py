
import numpy as np
import os
import pathlib
import sys
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader

from docopts.help_segmentation_test import parse_args
from frameworks.pytorch.loggers.local_logger import LocalLogger
from frameworks.pytorch.losses.dice_loss import DiceLoss
from frameworks.pytorch.losses.focal_loss import FocalLoss
from frameworks.pytorch.metrics.mIoU import torch_mIoU
from frameworks.pytorch.utils.scales_process import torch_batch_scales_process_numpy
from platges_utils.dataloader.collate_fn.segmentation_collate import build_segmentation_test_collate
from platges_utils.datasets.argusNL_dataset import ArgusNLDataset
from platges_utils.datasets.augmented_datasets import build_test_dataset
from platges_utils.datasets.platgesbcn_segmentation_dataset import PlatgesBCNSegmentationDataset, resolve_ambiguity_platgesBCN
from platges_utils.model.pyconvsegnet import build_PyConvSegNet_from_params
from visualization_utils.save_rgb_seg import save_rgb_seg, save_rgb_err, save_rgb_ovr_seg, save_rgb_ovr_err


WATER_ID = 1
SAND_ID = 2
OTHERS_ID = 0
CLASSES_COLOR = {
    WATER_ID : 'BLUE',
    SAND_ID : 'YELLOW',
    OTHERS_ID : 'GRAY'
}

EXPERIMENT_TYPE = "segments_argusNL"
DATA_ROOT = "./data_lists/"
experiment_metadata = SimpleNamespace(
    dataset             = 'argusNL',
    loss_name           = 'dice_loss',
    model_name          = 'pyConvSegNet',
    list_path           = './data_lists/argusNL_all.csv',
    output_root         = f"./outputs/{EXPERIMENT_TYPE}/",
    segmentation_prefix = 'seg_',
    error_prefix        = 'err_',
    overlapped_prefix   = 'ovr_'
)

MEAN = [0.485, 0.456, 0.406] # PyConvSegNet values
STD = [0.229, 0.224, 0.225] # PyConvSegNet values
VALUE_SCALE = 255
params = SimpleNamespace(
    # Parameters updated from data
    mean = [item * VALUE_SCALE for item in MEAN],
    std = [item * VALUE_SCALE for item in STD],
    # Modifiable hyperparams:
    # (add new hyperparameters for any optim, loss, etc)
    num_classes = 3,
    gamma = 2, # if focal_loss
    stride_rate = 2/3,
    scales = [1.0],
    # Data Augmentation parameters:
    resize_height = 512,
    resize_width = 696,
    crop_height = 473,
    crop_width = 473,
    # PyConvSegNet hyperparams:
    funnel_map = True,
    zoom_factor = 8,
    layers = 152,
    num_classes_pretrain = 150,
    backbone_output_stride = 8,
    backbone_net = "pyconvresnet",
    # Pre-trained pathes
    pretrained_back_path = None,
    adapt_state_dict = False,
    pretrained_path = None
)

####################################

def unNormalize(tensor, mean, std):
    # tensor: (C, H, W)
    for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    return tensor

def save_result(input_, output, target, img_path, experiment_metadata, params, classes_color=CLASSES_COLOR):
    # input_ from torch standard (C, H, W)
    # output as image mask (H, W)
    # target as image mask (H, W)

    input_ = unNormalize(input_, params.mean, params.std)

    output = output.numpy().astype(np.uint8)
    target = target.numpy().astype(np.uint8)
    input_ = (input_.permute(1, 2, 0).numpy() * 256).astype(np.uint8)

    folder = experiment_metadata.output_root
    seg_prefix = experiment_metadata.segmentation_prefix
    err_prefix = experiment_metadata.error_prefix
    ovr_prefix = experiment_metadata.overlapped_prefix

    # Save images
    save_rgb_seg(f"{folder}/{seg_prefix}{os.path.basename(img_path)}", output, classes_color)
    save_rgb_err(f"{folder}/{err_prefix}{os.path.basename(img_path)}", output, target)
    save_rgb_ovr_seg(f"{folder}/{ovr_prefix}{seg_prefix}{os.path.basename(img_path)}", input_, output, classes_color)
    save_rgb_ovr_err(f"{folder}/{ovr_prefix}{err_prefix}{os.path.basename(img_path)}", input_, output, target, classes_color)

def get_postprocess_output_and_target_funct(dataset):
    if dataset == 'platgesBCN':
        def postprocess_platges(output, target):
            output, target = resolve_ambiguity_platgesBCN(output, target, resolve=False) # output is B, H, W
            return output, target # No need output.max(1)[1]
    #elif dataset == '':
    else:
        return None

def get_base_dataset_type(dataset):
    if dataset == 'argusNL':
        return ArgusNLDataset
    elif dataset == 'platgesBCN':
        return PlatgesBCNSegmentationDataset
    else:
        raise Exception(f"Undefined dataset: {dataset}\nMaybe it is defined but not contemplated on the script (experiment).")

def get_model_type(model_name):
    if model_name == 'pyConvSegNet':
        return build_PyConvSegNet_from_params
    #elif model_name == :
    else:
        raise Exception(f"Undefined model_name: {model_name}\nMaybe it is defined but not contemplated on the script (experiment).")

def main(experiment_metadata, params, device, metrics_funct_dict=None):
    device = device or torch.device('cpu')

    base_dataset_type = get_base_dataset_type(experiment_metadata.dataset)
    dataset = base_dataset_type(experiment_metadata.list_path)
    dataset = build_test_dataset(dataset, params.resize_height, params.resize_width, params.mean, params.std)

    collate_fn = build_segmentation_test_collate(device=device)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    model_type = get_model_type(experiment_metadata.model_name)
    model = model_type(params)
    # model.eval() # <- When mode evaluation (no dropout, fixed batchnorm2d), all becomes 0; I don't know why

    criterion = lambda output, target : 1234

    metrics_funct_dict = metrics_funct_dict or {'mIoU' : torch_mIoU} # model applier torch_batch_scales_process_numpy already has argmax(C)
    local_logger = LocalLogger(metrics_funct_dict.copy(), len(dataset), prefix="Test")

    def model_applier(batch_images):
        output = torch_batch_scales_process_numpy(model, batch_images, params.num_classes, params.crop_height, params.crop_width, params.mean, params.std, params.scales, base_size=0, stride_rate=params.stride_rate, device=device)
        return torch.tensor(np.array(output)) # It's a B, H, W tensor mask

    postprocess_output_and_target_funct = get_postprocess_output_and_target_funct(experiment_metadata.dataset)

    with torch.no_grad():
        for input_, target, img_pathes in dataloader:
            output = model_applier(input_) # Returns B, H, W tensor masks

            if postprocess_output_and_target_funct is not None:
                output, target = postprocess_output_and_target_funct(output, target)
            
            loss = criterion(output, target)
            local_logger.update_epoch_log(output, target, loss, VERBOSE=True)

            for i, img_path in enumerate(img_pathes):
                save_result(input_[i], output[i], target[i], img_path, experiment_metadata, params)
        
        local_logger.finish_epoch(VERBOSE=True)

####################################


if __name__ == "__main__":
    args_experiment_metadata, args_params = parse_args(sys.argv)
    
    device = torch.device('cpu')

    if args_experiment_metadata == None or args_params == None:
        pathlib.Path(experiment_metadata.output_root).mkdir(parents=True, exist_ok=True)

        main(experiment_metadata, params, device, metrics_funct_dict=None)
    else:
        args_params.crop_height = min((args_params.resize_height // 8) * 8 + 1, args_params.crop_height) # Meh
        args_params.crop_width = min((args_params.resize_width // 8) * 8 + 1, args_params.crop_width) # Meh

        pathlib.Path(args_experiment_metadata.output_root).mkdir(parents=True, exist_ok=True)

        main(args_experiment_metadata, args_params, device, metrics_funct_dict=None)
