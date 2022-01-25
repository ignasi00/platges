
import albumentations as A
import albumentations.pytorch
import cv2
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets.argusNL_dataset import ArgusNLDataset
from datasets.wrapping_datasets.transforms_dataset import TransformDataset
from datasets.wrapping_datasets.dataset_specific.argusNL_to_platges_dataset import ArgusNL_to_PlatgesDataset
from loggers.wandb import WandB_logger
from losses.dice_loss import DiceLoss
from metrics.mIoU import torch_mIoU
from train_utils.early_stop.validation_epoch_divergence_stop import ValidationEpochDivergenceStop

from extern.pyconvsegnet.model.pyconvsegnet import PyConvSegNet
from extern.pyConvSegNet_utils import apply_net_train


def build_datasets(data_path, train_prob, default_value=-1):
    argusNL = ArgusNLDataset(data_path)
    platges = ArgusNL_to_PlatgesDataset(argusNL, default_value=default_value)

    idxs = list(range(len(platges)))
    random.shuffle(idxs)

    div = int(len(platges) * train_prob)

    train_dataset = Subset(platges, idxs[:div])
    val_dataset = Subset(platges, idxs[div:])

    return train_dataset, val_dataset

def build_train_transforms(resize_height, resize_width, crop_height, crop_width, mean=None, std=None, value_scale=255):
    mean = mean or [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = std or [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transforms = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.2, shift_limit=0.1, rotate_limit=10, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        A.Resize(resize_height, resize_width, interpolation=cv2.INTER_AREA, always_apply=True), 
        A.RandomCrop(crop_height, crop_width, p=1), #assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        A.Normalize(mean=mean, std=std),
        A.pytorch.transforms.ToTensorV2()
    ]

    return A.Compose(train_transforms)

def build_val_transforms(resize_height, resize_width, crop_height, crop_width, mean=None, std=None, value_scale=255):
    mean = mean or [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = std or [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transforms = [
        A.Resize(resize_height, resize_width, interpolation=cv2.INTER_AREA, always_apply=True),
        A.RandomCrop(crop_height, crop_width, p=1),
        A.Normalize(mean=mean, std=std),
        A.pytorch.transforms.ToTensorV2()
    ]

    return A.Compose(val_transforms)

def buid_dataloader(dataset, transforms, batch_size=1):
    dataset = TransformDataset(dataset, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size) # collate_fn, sampler, etc
    
    return dataloader

def build_model(layers, num_classes, zoom_factor, backbone_output_stride, backbone_net, pretrained_path=None):
    model = PyConvSegNet(layers=layers, classes=num_classes, zoom_factor=zoom_factor,
                                    pretrained=False, backbone_output_stride=backbone_output_stride,
                                    backbone_net=backbone_net)

    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path)
        checkpoint.pop('aux.4.weight')
        checkpoint.pop('aux.4.bias')
        checkpoint.pop('cls.1.weight')
        checkpoint.pop('cls.1.bias')
        model.load_state_dict(checkpoint, strict=False)

    return model

def build_loss(loss_type, *loss_params):
    return loss_type(*loss_params)

def build_optim(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def process_data(segmentation_net, img, train=True, cuda=False):
    output = apply_net_train(segmentation_net, img, train, cuda)
    return output

def compute_metrics(metrics_dict, seg_imgs, ground_truth, prefix):
    output = dict()
    for name, metric_func in metrics_dict.items():
        output[f'{prefix}{name}'] = metric_func(ground_truth, seg_imgs)
    return output

def optimization_step(optimizer, loss):
    optimizer.zero_grad() # clean gradient buffer from previous run
    loss.backward()
    optimizer.step()

def save_model(logger, model, epoch, filename, imgs):

    #torch.onnx.export(model, imgs[1], filename)
    torch.save(model.state_dict(), filename)
    # TODO: manage aliases like "best_mIoU", "best", etc
    logger.save_model(filename, aliases=[f"epoch_{epoch}"])


def main(data_path, train_prob, resize_height, resize_width, crop_height, crop_width, mean, std, value_scale, batch_size,
            layers, num_classes, zoom_factor, backbone_output_stride, backbone_net, pretrained_path, learning_rate, 
            experiment_name, entity, log_freq, num_epochs, val_epoch_freq, early_stop_memory, filename, cuda):
    
    train_dataset, val_dataset = build_datasets(data_path, train_prob, default_value=-1)

    train_transforms = build_train_transforms(resize_height, resize_width, crop_height, crop_width, mean=mean, std=std, value_scale=value_scale)
    train_dataloader = buid_dataloader(train_dataset, train_transforms, batch_size) # list or folder

    val_transforms = build_val_transforms(resize_height, resize_width, crop_height, crop_width, mean=mean, std=std, value_scale=value_scale)
    val_dataloader = buid_dataloader(val_dataset, val_transforms, batch_size=batch_size)

    segmentation_net = build_model(layers, num_classes, zoom_factor, backbone_output_stride, backbone_net, pretrained_path)

    # TODO: ¿construct objects including the model as initialization input (the loss will watch the model internally) for regularizations?
    loss_type = DiceLoss
    loss_params = [torch.mean]
    loss_function = build_loss(loss_type, *loss_params)
    # TODO: work on changing optimizers
    optimizer = build_optim(segmentation_net, learning_rate)

    metrics_dict = {'mIoU' : torch_mIoU}

    early_stoper = ValidationEpochDivergenceStop(epochs=early_stop_memory, criteria=min)

    MODEL = "argusNL_pyConvSegNet"
    config_summary = {
        "input_size (height, width)" : f"({resize_height}, {resize_width})", "crop_size" : f"({crop_height}, {crop_width})",
        "normalization (mean, std)" : f"({mean}, {std})", "value_scale" : value_scale, "zoom_factor" : zoom_factor,
        "batch_size" : batch_size, "learning_rate" : learning_rate, "model" : MODEL
    }

    project_name = "platgesBCN"
    logger = WandB_logger(project_name, experiment_name, entity)
    logger.watch_model(segmentation_net, log="all", log_freq=log_freq)
    logger.summary(config_summary)

    for epoch in range(1, num_epochs+1): # an epoch
        # minibatch_size = 1 => image update, minibatch_size = len(dataloader) => epoch update, in between => minibatch update
        # TODO: big minibatch size needs other strategies with less simultaneus memory usage
        for id_batch, (imgs, ground_truth, imgs_path) in enumerate(train_dataloader): # a minibatch

            if cuda: ground_truth = ground_truth.cuda(non_blocking=True)
            
            seg_imgs = process_data(segmentation_net, imgs, train=True, cuda=cuda)
            loss = loss_function(seg_imgs, ground_truth)

            metrics = compute_metrics(metrics_dict, seg_imgs, ground_truth, 'train_')
            metrics['train_loss'] = loss

            optimization_step(optimizer, loss)

            logger.log_metrics(metrics, step=None, commit=False)

            train_idx = (epoch - 1) * len(train_dataloader) + id_batch
            logger.log({'train_epoch' : epoch, 'train_batch' : id_batch, 'train_idx' : train_idx}, step=None, commit=True)
        
        save_model(logger, segmentation_net, epoch, filename, imgs)

        if epoch % val_epoch_freq == 0:
            for id_batch, (imgs, ground_truth, imgs_path) in enumerate(val_dataloader):

                if cuda: ground_truth = ground_truth.cuda(non_blocking=True)

                with torch.no_grad():
                    seg_imgs = process_data(segmentation_net, imgs, train=False, cuda=cuda)
                    loss = loss_function(seg_imgs, ground_truth)

                    metrics = compute_metrics(metrics_dict, seg_imgs, ground_truth, 'val_')
                    metrics['val_loss'] = loss
                
                logger.log_metrics(metrics, step=None, commit=False)

                val_idx = (epoch / val_epoch_freq - 1) * len(val_dataloader) + id_batch
                logger.log({'val_epoch' : epoch, 'val_batch' : id_batch, 'val_idx' : val_idx}, step=None, commit=True)
            
        epoch_log = logger.log_epoch(epoch, step=None, commit=True)
        
        try:
            if early_stoper.doStop(epoch_log['epoch_mean_val_loss']) : break
        except:
            continue

    # TODO: define a global summary (best epoch and value) for metrics that can be better high or better low



if __name__ == "__main__":

    from datetime import datetime


    data_path = '/home/ignasi/platges/data_lists/argusNL_train.csv'
    train_prob = 0.7
    resize_height = int(1024 / 2)
    resize_width = int(1392 / 2)
    crop_height = 473
    crop_width = 473
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    value_scale = 255
    batch_size = 2 #9
    layers = 152
    num_classes = 3
    zoom_factor = 8
    backbone_output_stride = 8
    backbone_net = "pyconvresnet"
    pretrained_path = '/home/ignasi/platges/extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth'
    learning_rate = 0.0001
    
    experiment_name = datetime.now().strftime("%Y%m%d_testsystem")
    entity = "ignasi00"
    log_freq = 5
    num_epochs = 30
    val_epoch_freq = 3
    early_stop_memory = 2
    #filename = "/home/ignasi/platges/model_parameters/testsystem_model.onnx"
    filename = "/home/ignasi/platges/model_parameters/testsystem_model.pth"

    cuda = False

    main(   data_path, train_prob, resize_height, 
            resize_width, crop_height, crop_width, 
            mean, std, value_scale, batch_size,
            layers, num_classes, zoom_factor, 
            backbone_output_stride, backbone_net, 
            pretrained_path, learning_rate, 
            experiment_name, entity, log_freq, num_epochs, 
            val_epoch_freq, early_stop_memory, filename, cuda)
