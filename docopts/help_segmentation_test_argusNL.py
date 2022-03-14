
from docopt import docopt
import json


#LIST_PATH = None
#OUTPUTS_ROOT = None
#MODEL_OUTPUTS_ROOT = None

MODEL_NAME = "pyConvSegNet"

# Params:
BATCH_SIZE = 1
RESIZE_HEIGHT = 512
RESIZE_WIDTH = 696
CROP_H = 473
CROP_W = 473
ZOOM_FACTOR = 8
BASE_SIZE = 512
SCALES = [1.0]

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
  segmentation_test_argusNL.py [--list_path=<lp>] [--outputs_root=<or>] [--models_root=<mr>] [--model_name=<mn>] [--batch_size=<bs>] [--resize_height=<rh>] [--resize_width=<rw>] [--crop_height=<ch>] [--crop_width=<cw>] [--zoom_factor=<zf>] [--base_size=<b>] [--scales=<s>...] [--funnel_map=<fm>] [--layers=<l>] [--num_classes_pretrain=<ncp>] [--backbone_output_stride=<bos>] [--backbone_net=<bn>] [--pretrained_backbone_path=<pbp>] [--pretrained_path=<pp>]
  segmentation_test_argusNL.py -h | --help

Options:
  -h --help                             Show this screen.
  --list_path=<lp>                      Str. List used to make the dataset.
  --outputs_root=<or>                   Str. Root folder where the outputs are stored.
  --models_root=<mr>                    Str. Root folder where the models are stored.
  --model_name=<mn>                     Str. Not implemented yet [default: {MODEL_NAME}].
  --batch_size=<bs>                     Int. Number of samples to be processed simultaneously (before applying a optimization step) [default: {BATCH_SIZE}].
  --resize_height=<rh>                  Int. Height of the full image on the output of the dataset/dataloader, it involves downsampling or interpolation [default: {RESIZE_HEIGHT}].
  --resize_width=<rw>                   Int. Width of the full image on the output of the dataset/dataloader, it involves downsampling or interpolation [default: {RESIZE_WIDTH}].
  --crop_height=<ch>                    Int. Height of the crop of the image at the input of the network [default: {CROP_H}].
  --crop_width=<cw>                     Int. Width of the crop of the image at the input of the network [default: {CROP_W}].
  --zoom_factor=<zf>                    Int. Zoom factor value for the models that needs it, it involves bilinear interpolation [default: {ZOOM_FACTOR}].
  --base_size=<b>                       Int. Base size value for the models that needs it [default: {BASE_SIZE}].
  --scales=<s>                          Float vector. Values for a multi-scale analisis, scores for different scales are added before thresholding [default: {' '.join([str(x) for x in SCALES])}].
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

    list_path = opts["--list_path"]
    outputs_root = opts["--outputs_root"]
    models_root = opts["--models_root"]
    model_name = opts["--model_name"]
    batch_size = int(opts["--batch_size"])
    resize_height = int(opts["--resize_height"])
    resize_width = int(opts["--resize_width"])
    crop_height = int(opts["--crop_height"])
    crop_width = int(opts["--crop_width"])
    zoom_factor = int(opts["--zoom_factor"])
    base_size = int(opts["--base_size"])
    scales = [float(x) for x in opts['--scales']]
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

    
    args = (list_path, outputs_root, models_root, model_name, batch_size, resize_height, resize_width, crop_height, crop_width, zoom_factor, base_size, scales, funnel_map, layers, num_classes_pretrain, backbone_output_stride, backbone_net, pretrained_backbone_path, pretrained_path)
    return args
