
import albumentations as A
import albumentations.pytorch
import cv2
import numpy as np
import os
import pathlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace

from datasets.argusNL_dataset import ArgusNLDataset
from datasets.wrapping_datasets.transforms_dataset import TransformDataset
from docopts.help_segmentation_test_argusNL import parse_args
from loggers.local_logger import LocalLogger
from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss
from metrics.mIoU import torch_mIoU
from rutines.validation.vanilla_validate import vanilla_validate
from visual_utils.save_rgb_seg import save_rgb_seg, save_rgb_err, save_rgb_ovr_seg, save_rgb_ovr_err

from extern.pyconvsegnet.model.pyconvsegnet import PyConvSegNet
from extern.pyConvSegNet_utils import apply_net_eval_cpu


SEGMENTATION_PREFIX = 'seg_'
ERROR_PREFIX = 'err_'
OVERLAPPED_PREFIX = 'ovr_'

#PROJECT_NAME = "platgesBCN"
#ENTITY = "ignasi00"
EXPERIMENT_TYPE = "segments_argusNL"

LIST_PATH = "./data_lists/argusNL_all.csv"
OUTPUTS_ROOT = f"./outputs/{EXPERIMENT_TYPE}/"
MODELS_ROOT = f"model_parameters/{EXPERIMENT_TYPE}/"

WATER_ID = 1
SAND_ID = 2
OTHERS_ID = 0
FUNNEL_MAP = {
    WATER_ID : [9, 21, 26, 37, 109, 113, 128],
    SAND_ID : [0, 46, 81, 94]
}

CLASSES_COLOR = {
    WATER_ID : 'BLUE',
    SAND_ID : 'YELLOW',
    OTHERS_ID : 'GRAY'
}


params = SimpleNamespace(
    # I/O params:
    list_path = f"{LIST_PATH}",
    outputs_root = f"{OUTPUTS_ROOT}",
    model_outputs_root = f"{MODELS_ROOT}",
    # Modifiable hyperparams:
    resize_height = 512,
    resize_width = 696,
    crop_h = 473,
    crop_w = 473,
    funnel_map = True,
    batch_size = 1,
    zoom_factor = 8,
    base_size = 512,
    scales = [1.0],
    # PyConvSegNet hyperparams:
    layers = 152,
    num_classes_pretrain=150,
    backbone_output_stride = 8,
    backbone_net = "pyconvresnet",
    # Pre-trained
    pretrained_back_path = None,
    pretrained_path = None
)


def build_dataset(list_path, resize_height, resize_width):
    argusNL_dataset = ArgusNLDataset(list_path)

    transforms_list = [
        A.Resize(resize_height, resize_width, interpolation=cv2.INTER_AREA, always_apply=True), #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        A.pytorch.transforms.ToTensorV2()
    ]
    transforms = A.Compose(transforms_list)

    argusNL_seg_dataset = TransformDataset(argusNL_dataset, transforms)
    return argusNL_seg_dataset

def build_model(layers, num_classes_pretrain, num_classes, zoom_factor, backbone_output_stride, backbone_net, funnel_map=None, pretrained_back_path=None, pretrained_path=None):
    backbone = PyConvSegNet(layers=layers, classes=num_classes_pretrain, zoom_factor=zoom_factor,
                                    pretrained=False, backbone_output_stride=backbone_output_stride,
                                    backbone_net=backbone_net)

    if pretrained_back_path is not None:
        checkpoint = torch.load(pretrained_back_path, map_location=torch.device('cpu'))
        backbone.load_state_dict(checkpoint, strict=False)

    head = nn.Conv2d(num_classes_pretrain, num_classes, kernel_size=1)
    model = nn.Sequential(backbone, head)

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint, strict=False)

    elif funnel_map is not None and isinstance(funnel_map, dict):
        head.weight.data = torch.zeros(head.weight.shape)
        for out, in_ in funnel_map.items():
            head.weight.data[out, :, 0, 0] += F.one_hot(torch.LongTensor(in_), num_classes=num_classes_pretrain).sum(dim=0).squeeze()
        head.weight.data[0, :, 0, 0] = (1 - head.weight.data[1:, :, 0, 0].sum(dim=0).squeeze()) / head.weight.data[1:, :, 0, 0].sum().squeeze()
        # head.weight.data[head.weight == 0] = torch.randn(head.weight[head.weight == 0].shape) * 1e-3

    return model

def build_model_applier(segmentation_net, num_classes, crop_h, crop_w, mean, std, base_size, scales):
    return lambda img : apply_net_eval_cpu(segmentation_net, img, num_classes, crop_h, crop_w, mean, std, base_size, scales)

def get_mean_and_std(dataset):
    # TODO: As a dataset is used, it should be enough to estmate them. On per sample inference, it should be a parameter obtained from all the training data.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std

def main(list_path, outputs_root, resize_height, resize_width, crop_h, crop_w, batch_size, layers, num_classes_pretrain, zoom_factor, base_size, scales, backbone_output_stride, backbone_net, pretrained_back_path, pretrained_path, funnel_map, VERBOSE_BATCH=True, VERBOSE_END=True):

    ####################################### PREPROCESSING  #######################################
    argusNL_seg_dataset = build_dataset(list_path, resize_height, resize_width)
    def collate_fn(batch):
        """
        for indices in batch_sampler:
            yield collate_fn([dataset[i] for i in indices])
        """
        return [ (torch.FloatTensor(input_), torch.IntTensor(target), dict(), img_path) for input_, target, _, img_path in batch ]
    argusNL_seg_dataloader = DataLoader(argusNL_seg_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = build_model(
        layers=layers,
        num_classes_pretrain=num_classes_pretrain,
        num_classes=3,
        zoom_factor=zoom_factor,
        backbone_output_stride=backbone_output_stride,
        backbone_net=backbone_net,
        funnel_map=funnel_map,
        pretrained_back_path=pretrained_back_path,
        pretrained_path=pretrained_path
    )

    mean, std = get_mean_and_std(argusNL_seg_dataset)
    model_applier = build_model_applier(model, 3, crop_h, crop_w, mean, std, base_size, scales)

    criterion = DiceLoss() # FocalLoss()

    metric_funct_dict = {'mIoU' : torch_mIoU}
    argusNL_seg_local_logger = LocalLogger(metric_funct_dict, len(argusNL_seg_dataset)) # TODO: logger that stores individual sample metrics and it is able to save them as csv
    ##############################################################################################

    #######################################   PROCESSING   #######################################
    argusNL_seg_local_logger.new_epoch()

    with torch.no_grad():
        for batch in argusNL_seg_dataloader:
            for input_, target, _, img_path in batch:
                # Dataloader should send its output to the requiered device and set the dtype (see collate_fn)

                output = model_applier(input_)
                target = target.unsqueeze(dim=0)
                output = output.unsqueeze(dim=0)
                loss = criterion(output.clone().float(), target) # Wrong because something => always 1
                
                # When python optimize the runtime, the logger update happens at the end of the epoch
                argusNL_seg_local_logger.update_epoch_log(output, target, loss, VERBOSE=VERBOSE_BATCH)

                output = output.squeeze().numpy()
                target = target.squeeze().numpy()
                input_ = input_.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)

                # Save images
                save_rgb_seg(f"{outputs_root}/{SEGMENTATION_PREFIX}{os.path.basename(img_path)}", output, CLASSES_COLOR)
                save_rgb_err(f"{outputs_root}/{ERROR_PREFIX}{os.path.basename(img_path)}", output, target)
                save_rgb_ovr_seg(f"{outputs_root}/{OVERLAPPED_PREFIX}{SEGMENTATION_PREFIX}{os.path.basename(img_path)}", input_, output, CLASSES_COLOR)
                save_rgb_ovr_err(f"{outputs_root}/{OVERLAPPED_PREFIX}{ERROR_PREFIX}{os.path.basename(img_path)}", input_, output, target, CLASSES_COLOR)

    argusNL_seg_local_logger.finish_epoch(VERBOSE=VERBOSE_END)
    ##############################################################################################

    ####################################### -------------- #######################################
    if VERBOSE_END : print(argusNL_seg_local_logger.get_last_epoch_log())
    ##############################################################################################


if __name__ == "__main__":

    # TODO: docopts
    (list_path, outputs_root, models_root, 
    model_name, batch_size, resize_height, 
    resize_width, crop_height, crop_width, 
    zoom_factor, base_size, scales, 
    funnel_map, layers, num_classes_pretrain, 
    backbone_output_stride, backbone_net, 
    pretrained_backbone_path, pretrained_path) = parse_args(sys.argv)


    params.pretrained_back_path = pretrained_backbone_path
    if params.pretrained_back_path is None and layers == 152 and num_classes_pretrain == 150 and backbone_output_stride == 8 and backbone_net == "pyconvresnet":
        params.pretrained_back_path = '/home/ignasi/platges/extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth'
    
    params.pretrained_path = pretrained_path
    
    params.funnel_map = FUNNEL_MAP if params.funnel_map == True else params.funnel_map
    params.list_path = list_path or params.list_path
    params.outputs_root = outputs_root or params.outputs_root
    params.resize_height = resize_height or params.resize_height
    params.resize_width = resize_width or params.resize_width
    params.crop_h = crop_height or params.crop_h
    params.crop_w = crop_width or params.crop_w
    params.batch_size = batch_size or params.batch_size
    params.layers = layers or params.layers
    params.num_classes_pretrain = num_classes_pretrain or params.num_classes_pretrain
    params.zoom_factor = zoom_factor or params.zoom_factor
    params.base_size = base_size or params.base_size
    params.scales = scales or params.scales
    params.backbone_output_stride = backbone_output_stride or params.backbone_output_stride
    params.backbone_net = backbone_net or params.backbone_net


    pathlib.Path(params.outputs_root).mkdir(parents=True, exist_ok=True)

    main(
        params.list_path,
        params.outputs_root,
        params.resize_height,
        params.resize_width,
        params.crop_h,
        params.crop_w,
        params.batch_size,
        params.layers,
        params.num_classes_pretrain,
        params.zoom_factor,
        params.base_size,
        params.scales,
        params.backbone_output_stride,
        params.backbone_net,
        params.pretrained_back_path,
        params.pretrained_path,
        params.funnel_map,
        VERBOSE_BATCH=True, VERBOSE_END=True
    )
