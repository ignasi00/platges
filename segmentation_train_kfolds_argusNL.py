
import albumentations as A
import albumentations.pytorch
import cv2
from datetime import datetime
from itertools import combinations # Or from more_itertools import distinct_combinations
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
from docopts.help_segmentation_train_kfolds_argusNL import parse_args
from loggers.local_logger import LocalLogger
from loggers.wandb_logger import WandbLogger
from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss
from metrics.mIoU import torch_mIoU
from rutines.training.accumulated_grad_train import accumulated_grad_train
from rutines.validation.vanilla_validate import vanilla_validate

from extern.pyconvsegnet.model.pyconvsegnet import PyConvSegNet


DATASET = 'argusNL'
MAX_BATCH_SIZE = 2

SEGMENTATION_PREFIX = 'seg_'
ERROR_PREFIX = 'err_'
OVERLAPPED_PREFIX = 'ovr_'

PROJECT_NAME = "platgesBCN"
ENTITY = "ignasi00"
EXPERIMENT_TYPE = "segments_argusNL"

LISTS_PATH = ["./data_lists/argusNL_K1-5.csv", "./data_lists/argusNL_K2-5.csv", "./data_lists/argusNL_K3-5.csv", "./data_lists/argusNL_K4-5.csv", "./data_lists/argusNL_K5-5.csv"]
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
    lists_path = LISTS_PATH,
    num_val_folds=1,
    outputs_root = f"{OUTPUTS_ROOT}",
    models_path = f"{MODELS_ROOT}/{EXPERIMENT_TYPE}.pt",
    experiment_name = datetime.now().strftime(f"%Y%m%d%H%M%S_{EXPERIMENT_TYPE}"),
    entity = ENTITY,
    project_name = PROJECT_NAME,
    # Modifiable hyperparams:
    num_epochs = 30,
    learning_rate = 0.0001,
    batch_size = 1,
    resize_height = 512,
    resize_width = 696,
    crop_h = 473,
    crop_w = 473,
    scale_limit = 0.2,
    shift_limit = 0.1,
    rotate_limit = 10,
    funnel_map = True,
    zoom_factor = 8,
    # PyConvSegNet hyperparams:
    layers = 152,
    num_classes_pretrain=150,
    backbone_output_stride = 8,
    backbone_net = "pyconvresnet",
    # Pre-trained
    pretrained_back_path = None,
    pretrained_path = None
)


def build_concat_dataset(lists_path):
    argusNL_dataset = []
    for list_path in lists_path:
        argusNL_dataset.append(ArgusNLDataset(list_path))
    
    if len(argusNL_dataset) == 1 : argusNL_dataset = argusNL_dataset[0]
    else : argusNL_dataset = torch.utils.data.ConcatDataset(argusNL_dataset)

    return argusNL_dataset

def build_train_dataset(lists_path, resize_height, resize_width, crop_height, crop_width, mean, std, scale_limit, shift_limit, rotate_limit):
    argusNL_dataset = build_concat_dataset(lists_path)

    transforms_list = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=scale_limit, shift_limit=shift_limit, rotate_limit=rotate_limit, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.Resize(resize_height, resize_width, interpolation=cv2.INTER_AREA, always_apply=True), 
        A.RandomCrop(crop_height, crop_width, p=1), #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        A.Normalize(mean=mean, std=std),
        A.pytorch.transforms.ToTensorV2()
    ]
    transforms = A.Compose(transforms_list)

    argusNL_seg_dataset = TransformDataset(argusNL_dataset, transforms, drop_extra_params=True)
    return argusNL_seg_dataset

def build_val_dataset(lists_path, resize_height, resize_width, crop_height, crop_width, mean, std):
    argusNL_dataset = build_concat_dataset(lists_path)

    transforms_list = [
        A.Resize(resize_height, resize_width, interpolation=cv2.INTER_AREA, always_apply=True),
        A.RandomCrop(crop_height, crop_width, p=1),
        A.Normalize(mean=mean, std=std),
        A.pytorch.transforms.ToTensorV2()
    ]
    transforms = A.Compose(transforms_list)

    argusNL_seg_dataset = TransformDataset(argusNL_dataset, transforms, drop_extra_params=True)
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
        head.weight.data[head.weight == 0] = torch.randn(head.weight[head.weight == 0].shape) * 1e-3

    return model

def get_mean_and_std(dataset, value_scale=255):
    # TODO: As a dataset is used, it should be enough to estmate them. On per sample inference, it should be a parameter obtained from all the training data.
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std

def one_fold(experiment_name, project_name, entity, argusNL_seg_train_dataset, argusNL_seg_val_dataset, outputs_root, models_path, resize_height, resize_width, crop_h, crop_w, mean, std, scale_limit, shift_limit, rotate_limit, batch_size, learning_rate, num_epochs, layers, num_classes_pretrain, zoom_factor, backbone_output_stride, backbone_net, pretrained_back_path, pretrained_path, funnel_map, fold, num_folds, device=None, model_name=None, save_epoch=False, VERBOSE_BATCH=True, VERBOSE_END=True):
    def collate_fn(batch):
        inputs = []
        targets = []

        for input_, target in batch:
            inputs.append(torch.FloatTensor(input_))
            targets.append(torch.IntTensor(target))

        inputs = torch.stack(inputs)
        inputs = inputs.to(device)
        targets = torch.stack(targets)
        targets = targets.to(device)

        return inputs, targets

    argusNL_seg_train_dataloader = DataLoader(argusNL_seg_train_dataset, batch_size=min(batch_size, MAX_BATCH_SIZE), collate_fn=collate_fn)
    argusNL_seg_val_dataloader = DataLoader(argusNL_seg_val_dataset, batch_size=min(batch_size, MAX_BATCH_SIZE), collate_fn=collate_fn)

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
    ).to(device)

    criterion = DiceLoss() # FocalLoss() # Crossentropy # TODO: selector

    optimizer_class = torch.optim.Adam
    optimizer_params = dict(lr=learning_rate, betas=(0.9, 0.999), weight_decay=0)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    metric_funct_dict = {'mIoU' : lambda seg_img, ground_truth : torch_mIoU(seg_img.argmax(dim=1), ground_truth)}
    argusNL_seg_train_local_logger = LocalLogger(metric_funct_dict, len(argusNL_seg_train_dataset), prefix=f"Train (fold {fold + 1}/{num_folds})")
    argusNL_seg_val_local_logger = LocalLogger(metric_funct_dict.copy(), len(argusNL_seg_val_dataset), prefix=f"Valid (fold {fold + 1}/{num_folds})")

    wandb_logger = WandbLogger(project_name, experiment_name, entity)
    wandb_logger.watch_model(model, log="all", log_freq=50)

    hyperparameters = {
        "resize_height" : resize_height,
        'resize_width' : resize_width,
        'crop_height' : crop_height,
        'crop_width' : crop_width,
        'mean' : '(' + ', '.join([str(x) for x in mean])[:-2] + ')',
        'std' : '(' + ', '.join([str(x) for x in std])[:-2] + ')',
        'scale_limit' : scale_limit,
        'shift_limit' : shift_limit,
        'rotate_limit' : rotate_limit,
        'batch_size' : batch_size,
        'num_epochs' : num_epochs,
        'learning_rate' : learning_rate,
        'funnel_map' : isinstance(funnel_map, dict),
        'zoom_factor' : zoom_factor,
        'layers' : layers,
        'num_classes_pretrain' : num_classes_pretrain,
        'backbone_output_stride' : backbone_output_stride,
        'backbone_net' : backbone_net,
        'model_name' : model_name,
        'num_folds' : num_folds,
        'dataset' : DATASET
        }
    hyperparameters['loss_type'] = type(criterion)
    hyperparameters['optim_type'] = type(optimizer)
    wandb_logger.summarize(hyperparameters)
    ##############################################################################################

    #######################################   PROCESSING   #######################################
    for epoch in range(num_epochs): # TODO: load from epoch => offset where needed (by instance, on best_epoch)
        
        model.eval()
        for m in model.modules(): # model.modules() yield all modules recursively (it goes inside submodules)
            if m.__class__.__name__.startswith('Dropout') or m.__class__.__name__.startswith('BatchNorm'):
                m.train()

        #last_train_epoch_log = accumulated_grad_train(model, criterion, optimizer, argusNL_seg_train_dataloader, argusNL_seg_train_local_logger, batch_size, device=device, drop_last=True, VERBOSE_BATCH=VERBOSE_BATCH, VERBOSE_END=VERBOSE_END)
        last_train_epoch_log = accumulated_grad_train(model, criterion, optimizer, argusNL_seg_train_dataloader, argusNL_seg_train_local_logger, batch_size, drop_last=True, VERBOSE_BATCH=VERBOSE_BATCH, VERBOSE_END=VERBOSE_END)
        wandb_logger.log(last_train_epoch_log, prefix=f"train_{fold + 1}-{num_folds}_", commit=False)
        argusNL_seg_train_local_logger.print_last_epoch_summary(mode='train')

        model.train()
        model.eval()

        with torch.no_grad():
            #last_val_epoch_log = vanilla_validate(model, criterion, argusNL_seg_val_dataloader, argusNL_seg_val_local_logger, device=device, VERBOSE_BATCH=VERBOSE_BATCH, VERBOSE_END=VERBOSE_END)
            last_val_epoch_log = vanilla_validate(model, criterion, argusNL_seg_val_dataloader, argusNL_seg_val_local_logger, VERBOSE_BATCH=VERBOSE_BATCH, VERBOSE_END=VERBOSE_END)
            wandb_logger.log(last_val_epoch_log, prefix=f"valid_{fold + 1}-{num_folds}_", commit=False)
            argusNL_seg_val_local_logger.print_last_epoch_summary(mode='valid')
        
        # Save model
        torch.save(model.state_dict(), models_path)
        wandb_logger.upload_model(models_path, aliases=[f'epoch_{epoch}'], wait=(epoch==(num_epochs-1)))
        
        wandb_logger.log({'epoch' : epoch})

    ##############################################################################################

    ####################################### -------------- #######################################
    best_epoch = argusNL_seg_val_local_logger.best_epochs()[0]

    wandb_logger.update_model(f'epoch_{best_epoch}', ['best'])

    wandb_logger.summarize({'best_epoch' : best_epoch})
    wandb_logger.summarize(argusNL_seg_train_local_logger.get_one_epoch_log(best_epoch, prefix=f"train_{fold + 1}-{num_folds}_"))
    wandb_logger.summarize(argusNL_seg_val_local_logger.get_one_epoch_log(best_epoch, prefix=f"valid_{fold + 1}-{num_folds}_"))

    #model_path = wandb_logger.download_model(os.path.basename(models_path), os.path.dirname(models_path), alias='best')
    #model.load_state_dict(torch.load(model_path))
    ##############################################################################################

    ##############################################################################################
    best_train = argusNL_seg_train_local_logger.get_one_epoch_log(best_epoch, prefix=f"train_")
    best_valid = argusNL_seg_val_local_logger.get_one_epoch_log(best_epoch, prefix=f"valid_")
    return best_train, best_valid
    ##############################################################################################


def main(experiment_name, project_name, entity, lists_path, num_val_folds, outputs_root, models_path, resize_height, resize_width, crop_h, crop_w, scale_limit, shift_limit, rotate_limit, batch_size, learning_rate, num_epochs, layers, num_classes_pretrain, zoom_factor, backbone_output_stride, backbone_net, pretrained_back_path, pretrained_path, funnel_map, device=None, model_name=None, VERBOSE_BATCH=True, VERBOSE_END=True):
    device = device or torch.device('cpu')

    ####################################### PREPROCESSING  #######################################
    mean, std = get_mean_and_std(build_concat_dataset(lists_path), value_scale=255)

    v_best_train = []
    v_best_valid = []

    folds = set(combinations(range(len(lists_path)), num_val_folds))
    num_folds = len(folds)
    for fold, indxs in enumerate(folds):

        experiment_fold_name = f'{experiment_name}_{fold + 1}-{num_folds}'

        train_lists = [lists_path[indx] for indx in range(len(lists_path)) if indx not in indxs]
        argusNL_seg_train_dataset = build_train_dataset(train_lists, resize_height, resize_width, crop_h, crop_w, mean, std, scale_limit, shift_limit, rotate_limit)

        val_lists = [lists_path[indx] for indx in indxs]
        argusNL_seg_val_dataset = build_val_dataset(val_lists, resize_height, resize_width, crop_h, crop_w, mean, std)

        best_train, best_valid = one_fold(
            experiment_fold_name, project_name, entity, 
            argusNL_seg_train_dataset, argusNL_seg_val_dataset,
            outputs_root, models_path, 
            resize_height, resize_width, crop_h, crop_w, mean, std,
            scale_limit, shift_limit, rotate_limit, 
            batch_size, learning_rate, num_epochs, 
            layers, num_classes_pretrain, zoom_factor, 
            backbone_output_stride, backbone_net, 
            pretrained_back_path, pretrained_path, 
            funnel_map, fold, num_folds, device=device,
            model_name=model_name, save_epoch=fold==0,
            VERBOSE_BATCH=VERBOSE_BATCH, VERBOSE_END=VERBOSE_END
            )

        v_best_train.append(best_train)
        v_best_valid.append(best_valid)

    #best_fold = [elem['valid_mIoU'] for elem in v_best_valid]
    #best_fold = sorted([(i, score) for i, score in enumerate(best_fold)], key=lambda x : x[-1], reverse=True)[0][0]

    wandb_logger = WandbLogger(project_name, experiment_name, entity)
    #wandb_logger.summarize(v_best_train[best_fold])
    #wandb_logger.summarize(v_best_valid[best_fold])
    wandb_logger.summarize(sum([elem['train_mIoU'] for v_best_train]) / len(v_best_train))
    wandb_logger.summarize(sum([elem['valid_mIoU'] for v_best_valid]) / len(v_best_valid))


if __name__ == "__main__":
    
    # TODO: docopts
    (lists_path, num_val_folds, outputs_root, models_path, 
    model_name, batch_size, learning_rate, num_epochs,
    resize_height, resize_width, crop_height, crop_width, 
    scale_limit, shift_limit, rotate_limit, 
    zoom_factor, funnel_map, layers, num_classes_pretrain, 
    backbone_output_stride, backbone_net, 
    pretrained_backbone_path, pretrained_path) = parse_args(sys.argv)


    crop_height = min((resize_height // 8) * 8 + 1, crop_height)
    crop_width = min((resize_width // 8) * 8 + 1, crop_width)

    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("WARNING: Training without GPU can be very slow!")


    params.pretrained_back_path = pretrained_backbone_path
    if params.pretrained_back_path is None and layers == 152 and num_classes_pretrain == 150 and backbone_output_stride == 8 and backbone_net == "pyconvresnet":
        params.pretrained_back_path = './extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth'
    
    params.pretrained_path = pretrained_path
    
    params.funnel_map = FUNNEL_MAP if params.funnel_map == True else params.funnel_map
    params.lists_path = lists_path or params.lists_path
    params.num_val_folds = num_val_folds or params.num_val_folds
    params.outputs_root = outputs_root or params.outputs_root
    params.models_path = models_path or params.models_path
    params.resize_height = resize_height or params.resize_height
    params.resize_width = resize_width or params.resize_width
    params.crop_h = crop_height or params.crop_h
    params.crop_w = crop_width or params.crop_w
    params.scale_limit = scale_limit or params.scale_limit
    params.shift_limit = shift_limit or params.shift_limit
    params.rotate_limit = rotate_limit or params.rotate_limit
    params.batch_size = batch_size or params.batch_size
    params.learning_rate = learning_rate or params.learning_rate
    params.num_epochs = num_epochs or params.num_epochs
    params.layers = layers or params.layers
    params.num_classes_pretrain = num_classes_pretrain or params.num_classes_pretrain
    params.zoom_factor = zoom_factor or params.zoom_factor
    params.backbone_output_stride = backbone_output_stride or params.backbone_output_stride
    params.backbone_net = backbone_net or params.backbone_net


    pathlib.Path(params.outputs_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(params.models_path)).mkdir(parents=True, exist_ok=True)

    main(
        params.experiment_name,
        params.project_name,
        params.entity,
        params.lists_path,
        params.num_val_folds,
        params.outputs_root,
        params.models_path,
        params.resize_height,
        params.resize_width,
        params.crop_h,
        params.crop_w,
        params.scale_limit,
        params.shift_limit,
        params.rotate_limit,
        params.batch_size,
        params.learning_rate,
        params.num_epochs,
        params.layers,
        params.num_classes_pretrain,
        params.zoom_factor,
        params.backbone_output_stride,
        params.backbone_net,
        params.pretrained_back_path,
        params.pretrained_path,
        params.funnel_map,
        device=device,
        model_name=model_name,
        VERBOSE_BATCH=True, VERBOSE_END=True
    )
