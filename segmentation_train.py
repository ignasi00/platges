
from datetime import datetime
from itertools import combinations # Or from more_itertools import distinct_combinations
import os
import pathlib
import sys
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader

from docopts.help_segmentation_train import parse_args
from framework.pytorch.datasets.utils.build_kfolds_datasets_generators import build_kfolds_datasets_generators
from framework.pytorch.loggers.kfolds_local_logger import KfoldsLocalLogger
from framework.pytorch.loggers.local_logger import LocalLogger
from framework.pytorch.losses.dice_loss import DiceLoss
from framework.pytorch.losses.focal_loss import FocalLoss
from framework.pytorch.metrics.mIoU import torch_mIoU
from platges_utils.dataloader.collate_fn.segmentation_collate import build_segmentation_collate
from platges_utils.datasets.argusNL_dataset import ArgusNLDataset
from platges_utils.datasets.augmented_datasets import build_train_dataset, build_val_dataset
from platges_utils.datasets.build_kfolds_datasets_iterators import build_kfolds_datasets_iterators
from platges_utils.datasets.concat_dataset import build_concat_dataset
from platges_utils.datasets.platgesbcn_segmentation_dataset import PlatgesBCNSegmentationDataset, resolve_ambiguity_platgesBCN
from platges_utils.logger.wandb_logger import WandbFinalSummarize, build_wandb_logger
from platges_utils.model.pyconvsegnet import build_PyConvSegNet_from_params
from platges_utils.rutines.training.accumulated_grad_train import accumulated_grad_train
from platges_utils.rutines.training.vanilla_train import vanilla_train
from platges_utils.rutines.validation.vanilla_validate import vanilla_validate
from platges_utils.rutines.epochs_managers import default_epochs
from platges_utils.rutines.kfolds_manager import kfolds_manager


MAX_BATCH_SIZE = 2

EXPERIMENT_TYPE = "segments_argusNL"
DATA_ROOT = "./data_lists/"
experiment_metadata = SimpleNamespace(
    experiment_type = EXPERIMENT_TYPE,
    dataset         = 'argusNL',
    training_type   = 'kfolds',
    loss_name       = 'dice_loss',
    optim_name      = 'Adam',
    model_name      = 'pyConvSegNet',
    lists_paths     = [f"{DATA_ROOT}/argusNL_K{i}-5.csv" for i in range(1, 5+1)],
    num_val_folds   = 1,
    num_folds       = 1,
    output_root     = f"./outputs/{EXPERIMENT_TYPE}/",
    models_root     = f"model_parameters/{EXPERIMENT_TYPE}/",
    experiment_name = datetime.now().strftime(f"%Y%m%d%H%M%S_{EXPERIMENT_TYPE}"),
    initial_epoch   = 0,
    project_name    = 'platgesBCN',
    entity          = 'ignasi00'
)

params = SimpleNamespace(
    # Parameters updated from data
    mean = 0,
    std = 1,
    # Modifiable hyperparams:
    # (add new hyperparameters for any optim, loss, etc)
    num_classes = 3,
    num_epochs = 30,
    learning_rate = 0.0001,
    batch_size = 1,
    gamma = 2, # if focal_loss
    betas = (0.9, 0.999), # if Adam optim
    weight_decay = 0,
    # Data Augmentation parameters:
    resize_height = 512,
    resize_width = 696,
    crop_height = 473,
    crop_width = 473,
    scale_limit = 0.2,
    shift_limit = 0.1,
    rotate_limit = 10,
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


# Each namespace has its own bacause later one may change and the other no
def experiment_metadata_add_elem(experiment_metadata, elem_name, elem)  : setattr(experiment_metadata, elem_name, elem)
def params_add_elem(params, elem_name, elem)                            : setattr(params, elem_name, elem)
def experiment_metadata_dict(experiment_metadata)                       : vars(experiment_metadata)
def params_dict(params)                                                 : vars(params)


####################################

def wandb_update_config(wandb_logger, experiment_metadata, params):
    # Both SimpleNamespace to dict and copy (if not, SimpleNamespace mutate), joint and update wandb
    summary = experiment_metadata_dict(experiment_metadata).copy()
    summary.update(params_dict(params))
    wandb_logger.summarize(summary)

def get_mean_and_std(dataset=None, params=None, value_scale=255):
    # TODO: As a dataset is used, it should be enough to estmate them. And for sample inference, it should be a parameter obtained from all the training data.
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std

def get_postprocess_output_and_target_funct(dataset):
    if dataset == 'platgesBCN':
        return resolve_ambiguity_platgesBCN
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

def get_loss_type(loss_name, params):
    if loss_name == 'dice_loss':
        build_dice_loss = lambda model : DiceLoss(reduce=None)
        return build_dice_loss
    elif loss_name == 'focal_loss':
        build_dice_loss = lambda model : FocalLoss(gamma=params.gamma ,reduce=None)
        return build_dice_loss
    else:
        raise Exception(f"Undefined loss_name: {loss_name}\nMaybe it is defined but not contemplated on the script (experiment).")

def get_optim_type(optim_name, params):
    if optim_name == 'Adam':
        build_Adam = lambda model : torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=params.betas, weight_decay=params.weight_decay)
        return build_Adam
    #elif optim_name == :
    else:
        raise Exception(f"Undefined optim_name: {optim_name}\nMaybe it is defined but not contemplated on the script (experiment).")

def copy_generator_from_params(constructor, params, num_folds):
    for _ in range(num_folds):
        yield constructor(params)

def main(experiment_metadata, params, device, max_batch_size=MAX_BATCH_SIZE, metrics_funct_dict=None):
    device = device or torch.device('cpu')

    base_dataset_type = get_base_dataset_type(experiment_metadata.dataset)
    model_type = get_model_type(experiment_metadata.model_name)
    loss_type = get_loss_type(experiment_metadata.loss_name, params)
    optim_type = get_optim_type(experiment_metadata.optim_name, params)

    postprocess_output_and_target_funct = get_postprocess_output_and_target_funct(experiment_metadata.dataset)

    mean, std = get_mean_and_std(dataset=build_concat_dataset(experiment_metadata.lists_paths, base_dataset_type), params=params, value_scale=255)
    params_add_elem(params, 'mean', mean)
    params_add_elem(params, 'std', std)

    metrics_funct_dict = metrics_funct_dict or {'mIoU' : lambda seg_img, ground_truth : torch_mIoU(seg_img.argmax(dim=1), ground_truth)}

    collate_fn = build_segmentation_collate(device=device)
    create_dataloader = lambda dataset : DataLoader(dataset, batch_size=min(params.batch_size, max_batch_size), collate_fn=collate_fn)

    models_path = f'{experiment_metadata.models_root}/{experiment_metadata.experiment_type}.pt'

    # Given that the inputs for the kfolds version are designed to be different, a big if clause is used although it increase the complexity (equivalent to 2 scripts).
    if experiment_metadata.training_type == "kfolds":
        folds = set(combinations(range(len(experiment_metadata.lists_paths)), experiment_metadata.num_val_folds))
        num_folds = len(folds)
        experiment_metadata_add_elem(experiment_metadata, 'num_folds', num_folds)

        train_dataset_iterator, val_dataset_iterator = build_kfolds_datasets_iterators(experiment_metadata.lists_paths, folds, base_dataset_type, params)
        
        model_generator = copy_generator_from_params(model_type, params, num_folds)
        model_iterator = iter(model_generator)
        
        kfolds_logger = KfoldsLocalLogger(metrics_funct_dict.copy(), num_folds, key=None, maximize=True, train_prefix="Train", val_prefix="Valid")
        
        wandb_exit_funct = WandbFinalSummarize(training_type=experiment_metadata.training_type, local_logger_list=[kfolds_logger], best_epoch_offset=experiment_metadata.initial_epoch, best_epochs_funct=lambda : kfolds_logger.get_results())
        
        with build_wandb_logger(experiment_metadata.project_name, experiment_metadata.experiment_name, experiment_metadata.entity, exit_funct=wandb_exit_funct) as wandb_logger:
            epochs_manager = lambda train_rutine, val_rutine, num_epochs, train_local_logger, val_local_logger, wandb_logger : default_epochs(train_rutine, val_rutine, num_epochs, train_local_logger, val_local_logger, wandb_logger, models_path=models_path)
            kfolds_manager(epochs_manager, params.num_epochs, train_dataset_iterator, val_dataset_iterator, create_dataloader, model_iterator, loss_type, optim_type, kfolds_logger, wandb_logger, postprocess_output_and_target_funct=postprocess_output_and_target_funct, max_batch_size=max_batch_size)
        
    elif experiment_metadata.training_type in ["vanilla"]:
        train_dataset = base_dataset_type(experiment_metadata.lists_paths[0])
        train_dataset = build_train_dataset(train_dataset, params.resize_height, params.resize_width, params.crop_height, params.crop_width, params.mean, params.std, params.scale_limit, params.shift_limit, params.rotate_limit)

        val_dataset = base_dataset_type(experiment_metadata.lists_paths[1])
        val_dataset = build_val_dataset(val_dataset, params.resize_height, params.resize_width, params.crop_height, params.crop_width, params.mean, params.std)

        train_dataloader = create_dataloader(train_dataset)
        val_dataloader = create_dataloader(val_dataset)

        model = model_type(params)
        criterion = loss_type(model)
        optim = optim_type(model)

        train_local_logger = LocalLogger(metrics_funct_dict.copy(), len(train_dataset), prefix="Train")
        val_local_logger = LocalLogger(metrics_funct_dict.copy(), len(val_dataset), prefix="Valid")

        wandb_exit_funct = WandbFinalSummarize(training_type=experiment_metadata.training_type, local_logger_list=[train_local_logger, val_local_logger], best_epoch_offset=experiment_metadata.initial_epoch, best_epochs_funct=lambda : val_local_logger.best_epochs()[0])
        
        with build_wandb_logger(experiment_metadata.project_name, experiment_metadata.experiment_name, experiment_metadata.entity, exit_funct=wandb_exit_funct) as wandb_logger:
            if min(params.batch_size, max_batch_size) == params.batch_size:
                train_rutine = lambda : vanilla_train(model, criterion, optim, train_dataloader, train_local_logger, postprocess_output_and_target_funct=postprocess_output_and_target_funct, VERBOSE_BATCH=True, VERBOSE_END=True)
            else:
                train_rutine = lambda : accumulated_grad_train(model, criterion, optim, train_dataloader, train_local_logger, postprocess_output_and_target_funct=postprocess_output_and_target_funct, drop_last=True, VERBOSE_BATCH=True, VERBOSE_END=True)
            val_rutine = lambda : vanilla_validate(model, criterion, val_dataloader, val_local_logger, postprocess_output_and_target_funct=postprocess_output_and_target_funct, VERBOSE_BATCH=True, VERBOSE_END=True)
            
            default_epochs(train_rutine, val_rutine, params.num_epochs, train_local_logger, val_local_logger, wandb_logger, epoch_offset=experiment_metadata.initial_epoch, models_path=models_path)
    else:
        raise Exception(f"Undefined training_type: {experiment_metadata.training_type}\nMaybe it is defined but not contemplated on the script (experiment).")

####################################


if __name__ == "__main__":
    args_experiment_metadata, args_params = parse_args(sys.argv)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("WARNING: Training without GPU can be very slow!")

    if args_experiment_metadata == None or args_params == None:
        pathlib.Path(experiment_metadata.output_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.dirname(experiment_metadata.models_root)).mkdir(parents=True, exist_ok=True)

        main(experiment_metadata, params, device, max_batch_size=MAX_BATCH_SIZE, metrics_funct_dict=None)
    else:
        args_params.crop_height = min((args_params.resize_height // 8) * 8 + 1, args_params.crop_height) # Meh
        args_params.crop_width = min((args_params.resize_width // 8) * 8 + 1, args_params.crop_width) # Meh

        pathlib.Path(args_experiment_metadata.output_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.dirname(args_experiment_metadata.models_root)).mkdir(parents=True, exist_ok=True)

        main(args_experiment_metadata, args_params, device, max_batch_size=MAX_BATCH_SIZE, metrics_funct_dict=None)
