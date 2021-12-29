
from docopt import docopt


VALUE_SCALE = 255


DATA_PATH = '/mnt/c/Users/Ignasi/Downloads/ArgusNL'
DOWNSAMPLE = 4

LAYERS = 152
CLASSES = 150
ZOOM_FACTOR = 8
BACKBONE_OUTPUT_STRIDE = 8
BACKBONE_NET = "pyconvresnet"
PRETRAINED_PATH = '/home/ignasi/platges/extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth'

CROP_H = 473
CROP_W = 473

MEAN = [0.485, 0.456, 0.406]
MEAN = [item * VALUE_SCALE for item in MEAN]
STD = [0.229, 0.224, 0.225]
STD = [item * VALUE_SCALE for item in STD]

BASE_SIZE = 512
SCALES = [1.0]

SAVE_PATH = '/mnt/c/Users/Ignasi/Downloads/argus_saved/'

COLORS_PATH = "extern/pyconvsegnet/dataset/ade20k/ade20k_colors.txt"
ALPHA = 0.5
ERROR_RED = True

VERBOSE = True


# Note: If a subcommand is repeated, the dictionary will have the number of times it appeared instead of a bool, but, the usages must be ordered from more restrictive to less restrictive
# TODO: [--option=<o> (<a> <b>) [<x> <y>]] or similar structures and then put the default values in function of the model (by instance, number of layers), dataset, etc
DOCTEXT = f"""
Usage:
  segmentation_test argus pyconvsegnet [ade20k] [--data_path=<dp>] [--downsample=<ds>] [--layers=<l>] [--classes=<c>] [--zoom_factor=<zf>] [--backbone_output_stride=<bos>] [--backbone=<b>] [--pretrained_path=<pp>] [--crop_h=<ch>] [--crop_w=<cw>] [--mean=<m>...] [--std=<s>...] [--base_size=<bs>] [--scales=<sc>...] [--save_dir=<sp>] [--color_path=<cp>] [--alpha=<a>] [--error_red] [--verbose]
  segmentation_test -h | --help


#####################################
segmentation_test <used dataset> <used model> [initial output mode] [options]

<used dataset> is a code name for the dataset or datasets that will be used for the test.
<used model> is a code name for the model name.
<initial output model> is a code name for the direct output of the model; when the output is not directly the our problem model, it will be somehow adapted

vector options like [--mean=<m>...] or [--std=<s>...] have to be inputed like: --mean=3 --mean=6 --mean=10 for a [3, 6, 10] output
#####################################


Options:
  -h --help                             Show this screen.
  --data_path=<dp>                      Set the folder with the source images [default: {DATA_PATH}].
  --downsample=<ds>                     Int. Set the downsample factor [default: {DOWNSAMPLE}].
  --layers=<l>                          Int. Number of network layers [default: {LAYERS}].
  --classes=<c>                         Int. Number of output classes the model have [default: {CLASSES}].
  --zoom_factor=<zf>                    Int. PyConvSegNet zoom factor [default: {ZOOM_FACTOR}].
  --backbone_output_stride=<bos>        Int. PyConvSegNet backbone output strice [default: {BACKBONE_OUTPUT_STRIDE}].
  --backbone=<b>                        Backbone of the model [default: {BACKBONE_NET}].
  --pretrained_path=<pp>                Path to the file where the models weights are saved [default: {PRETRAINED_PATH}].
  --crop_h=<ch>                         Int. PyConvSegNet crop H [default: {CROP_H}].
  --crop_w=<cw>                         Int. PyConvSegNet crop W [default: {CROP_W}].
  --mean=<m>                            Float vector. PyConvSegNet means [default: {' '.join([str(x) for x in MEAN])}].
  --std=<s>                             Float vector. PyConvSegNet standar deviations [default: {' '.join([str(x) for x in STD])}].
  --base_size=<bs>                      Int. PyConvSegNet base size [default: {BASE_SIZE}].
  --scales=<sc>                         Float vector. PyConvSegNet scales [default: {' '.join([str(x) for x in SCALES])}].
  --save_dir=<sd>                       Set the folder where the results will be stored [default: {SAVE_PATH}].
  --color_path=<cp>                     Path to the file where the colours for each class are defined [default: {COLORS_PATH}].
  --alpha=<a>                           Float. Factor to control the overlap of image and mask: 1 is all mask and 0 is all img [default: {ALPHA}].
  --error_red                           If set, the erroneous predicted regions will be red.
  --verbose                             If set, there will be text information about the progress.
"""

def process_mode(opts):
  if opts['argus'] and opts['pyconvsegnet']: return "argus_pyConvSegNet"

def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv, help=True, version=None, options_first=False)

    data_path = opts['--data_path']
    #
    save_path = opts['--save_dir']
    colors_path = opts['--color_path']
    alpha = float(opts['--alpha'])
    error_red = opts['--error_red']
    verbose = opts['--verbose']

    mode = process_mode(opts)

    args = []
    if mode == "argus_pyConvSegNet":
        downsample = int(opts['--downsample'])
        argus_labels = True
        layers = int(opts['--layers'])
        classes = int(opts['--classes'])
        ade20k_labels = True if opts['ade20k'] else False
        zoom_factor = int(opts['--zoom_factor'])
        backbone_output_stride = int(opts['--backbone_output_stride'])
        backbone_net = opts['--backbone']
        pretrained_path = opts['--pretrained_path']
        crop_h = int(opts['--crop_h'])
        crop_w = int(opts['--crop_w'])
        mean = [float(x) for x in opts['--mean']]
        std = [float(x) for x in opts['--std']]
        base_size = int(opts['--base_size'])
        scales = [float(x) for x in opts['--scales']]
        
        args = (data_path, downsample, argus_labels, layers, classes, ade20k_labels, zoom_factor, backbone_output_stride, backbone_net, pretrained_path, crop_h, crop_w, mean, std, base_size, scales, save_path, colors_path, alpha, error_red, verbose)
    
    return args, mode