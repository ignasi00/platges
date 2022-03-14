
from docopt import docopt
import json


#LIST_PATH_TRAIN = None
#LIST_PATH_VAL = None
#OUTPUTS_ROOT = None
#MODEL_OUTPUTS_ROOT = None
NUM_VAL_FOLDS = 1

MODEL_NAME = "pyConvSegNet"

# Params:
BATCH_SIZE = 1
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
RESIZE_HEIGHT = 512
RESIZE_WIDTH = 696
CROP_H = 473
CROP_W = 473
SCALE_LIMIT = 0.2
SHIFT_LIMIT = 0.1
ROTATE_LIMIT = 10
ZOOM_FACTOR = 8

FUNNEL_MAP = True

# PyConvSegNet params:
LAYERS = 152
NUM_CLASSES_PRETRAIN=150
BACKBONE_OUTPUT_STRIDE = 8
BACKBONE_NET = "pyconvresnet"
#PRETRAINED_BACK_PATH = None
#PRETRAINED_PATH = None


DOCTEXT = f"""
Usage:
  segmentation_train_kfolds_argusNL.py [--lists_path=<lp>...] [--num_val_folds=<nvf>] [--outputs_root=<or>] [--models_path=<mp>] [--model_name=<mn>] [--batch_size=<bs>] [--learning_rate=<lr>] [--num_epochs=<ne>] [--resize_height=<rh>] [--resize_width=<rw>] [--crop_height=<ch>] [--crop_width=<cw>] [--scale_limit=<sl>] [--shift_limit=<shl>] [--rotate_limit=<rl>] [--zoom_factor=<zf>] [--funnel_map=<fm>] [--layers=<l>] [--num_classes_pretrain=<ncp>] [--backbone_output_stride=<bos>] [--backbone_net=<bn>] [--pretrained_backbone_path=<pbp>] [--pretrained_path=<pp>]
  segmentation_train_kfolds_argusNL.py -h | --help

Options:
  -h --help                             Show this screen.
  --lists_path=<lp>                     Str vector. Lists used to make the datasets.
  --num_val_folds=<nvf>                 Int. Number of lists used for the validation datasets [default: {NUM_VAL_FOLDS}].
  --outputs_root=<or>                   Str. Root folder where the outputs are stored.
  --models_path=<mp>                    Str. Path and filename for model storage.
  --model_name=<mn>                     Str. Not implemented yet [default: {MODEL_NAME}].
  --batch_size=<bs>                     Int. Number of samples to be processed simultaneously (before applying a optimization step) [default: {BATCH_SIZE}].
  --learning_rate=<lr>                  Float. Positive float. It controlls the optimization speed and it is reallly important [default: {LEARNING_RATE}]. 
  --num_epochs=<ne>                     Int. Number of epochs (how many times the model see all the training data) [default: {NUM_EPOCHS}].
  --resize_height=<rh>                  Int. Height of the full image on the output of the dataset/dataloader, it involves downsampling or interpolation [default: {RESIZE_HEIGHT}].
  --resize_width=<rw>                   Int. Width of the full image on the output of the dataset/dataloader, it involves downsampling or interpolation [default: {RESIZE_WIDTH}].
  --crop_height=<ch>                    Int. Height of the crop of the image at the input of the network [default: {CROP_H}].
  --crop_width=<cw>                     Int. Width of the crop of the image at the input of the network [default: {CROP_W}].
  --scale_limit=<sl>                    Float. Scaling factor range [default: {SCALE_LIMIT}].
  --shift_limit=<shl>                   Float. Shift factor range for both height and width [default: {SHIFT_LIMIT}].
  --rotate_limit=<rl>                   Int. Positive. Maximum number of degrees to rotate in whatever direction [default: {ROTATE_LIMIT}].
  --zoom_factor=<zf>                    Int. Zoom factor value for the models that needs it, it involves bilinear interpolation [default: {ZOOM_FACTOR}].
  --funnel_map=<fm>                     Bool or str. If True, ade20k to platges initialization, if False random. Str should be a JSON dict (not list) path with keys from the (0, 1 and 2) set. [default: {FUNNEL_MAP}].
  --layers=<l>                          Int. Model param. Number of layers of the base network (without funneling layer) [default: {LAYERS}].
  --num_classes_pretrain=<ncp>          Int. Model param. Number of classes before the funneling them, in order to fully load a previous model [default: {NUM_CLASSES_PRETRAIN}].
  --backbone_output_stride=<bos>        Int. Model param. How times smaller is the output of the backbone of the base model in respect to the input image crop size [default: {BACKBONE_OUTPUT_STRIDE}].
  --backbone_net=<bn>                   Str. Model param. Name of the backbone network of the base model [default: {BACKBONE_NET}].
  --pretrained_backbone_path=<pbp>      Str. Model param. Path to the base network pretrained weights.
  --pretrained_path=<pp>                Str. Model param. Path to the full network pretrained weights.
  
"""



def parse_args(argv):
    
    opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    lists_path = opts["--lists_path"]
    num_val_folds = int(opts["--num_val_folds"])
    outputs_root = opts["--outputs_root"]
    models_path = opts["--models_path"]
    model_name = opts["--model_name"]
    batch_size = int(opts["--batch_size"])
    learning_rate = float(opts["--learning_rate"])
    num_epochs = int(opts["--num_epochs"])
    resize_height = int(opts["--resize_height"])
    resize_width = int(opts["--resize_width"])
    crop_height = int(opts["--crop_height"])
    crop_width = int(opts["--crop_width"])
    scale_limit = float(opts["--scale_limit"])
    shift_limit = float(opts["--shift_limit"])
    rotate_limit = int(opts["--rotate_limit"])
    zoom_factor = int(opts["--zoom_factor"])
    funnel_map = opts["--funnel_map"]
    layers = int(opts["--layers"])
    num_classes_pretrain = int(opts["--num_classes_pretrain"])
    backbone_output_stride = int(opts["--backbone_output_stride"])
    backbone_net = opts["--backbone_net"]
    pretrained_backbone_path = opts["--pretrained_backbone_path"]
    pretrained_path = opts["--pretrained_path"]


    if funnel_map == "True":
        funnel_map = True
    else:
        try:
            with open(funnel_map, 'r') as f:
                funnel_map = json.load(f)
        except:
            funnel_map = None

    
    args = (lists_path, num_val_folds, outputs_root, models_path, model_name, batch_size, learning_rate, num_epochs, resize_height, resize_width, crop_height, crop_width, scale_limit, shift_limit, rotate_limit, zoom_factor, funnel_map, layers, num_classes_pretrain, backbone_output_stride, backbone_net, pretrained_backbone_path, pretrained_path)
    return args
