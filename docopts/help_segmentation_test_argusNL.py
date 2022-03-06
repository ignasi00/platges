
from docopt import docopt
import json


LIST_PATH = None
OUTPUTS_ROOT = None
MODEL_OUTPUTS_ROOT = None

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
PRETRAINED_BACK_PATH = None
PRETRAINED_PATH = None


DOCTEXT = f"""
Usage:
  segmentation_test_argusNL.py [--list_path=<lp>] [--outputs_root=<or>] [--models_root=<mr>] 
  segmentation_test_argusNL.py -h | --help

Options:
  -h --help                             Show this screen.
  --list_path=<lp>
  --outputs_root=<or>
  --models_root=<mr>
  
"""



def parse_args(argv):
    
    opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)


     = opts["--kk"]

    
    args = ()
    return args
