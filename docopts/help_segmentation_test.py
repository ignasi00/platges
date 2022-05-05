
from docopt import docopt
from types import SimpleNamespace


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

MEAN = [0.4662, 0.5510, 0.5292] # Platges2021 values
STD = [0.2596, 0.1595, 0.1578] # Platges2021 values
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
    pretrained_back_path = 'model_parameters/original_pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth',
    adapt_state_dict = True,
    pretrained_path = None
)

DOCTEXT = f"""
Usage:
  segmentation_test.py [options] [--scales=<s>...] [--mean=<m>...] [--std=<s>...]
  segmentation_test.py default
  segmentation_test.py docopt_default
  segmentation_test.py -h | --help

No args, -h or --help will show this screen.
Mode default will use the values on segmentation_test.py SimpleNamespaces.
Setting at least 1 value or the "docopt_default" command will use the setted values and the defaults on this help page (from Docopts).

[--scales=<s>...] and similar parameters are assigned as --scales=1 --scales=1.25 --scales=0.75

Options:
  -h --help                         Show this screen.

  --dataset=<d>                     Str. Name of the dataset in use (argusNL, platgesBCN, etc) [default: {experiment_metadata.dataset}].
  --loss_name=<ln>                  Str. Name of the loss type to use (dice_loss, focal_loss, etc) [default: {experiment_metadata.loss_name}].
  --model_name=<mn>                 Str. Name of the model type to use (pyConvSegNet, etc) [default: {experiment_metadata.model_name}].
  --list_path=<lp>                  Str. Path to the list to be used [default: {experiment_metadata.list_path}].
  --output_root=<or>                Str. Path to the folder where the outputs will be stored [default: {experiment_metadata.output_root}].
  --segmentation_prefix=<sp>        Str. Prefix to generate the name of the detection image [default: {experiment_metadata.segmentation_prefix}].
  --error_prefix=ep>                Str. Prefix to generate the name of the error image [default: {experiment_metadata.error_prefix}].
  --overlapped_prefix=<op>          Str. Prefix to generate the name of the input image with the detection [default: {experiment_metadata.overlapped_prefix}].

  --mean=<m>                        Float vector. Population mean of each channel of the data [default: {' '.join([str(x) for x in params.mean])}].
  --std=<s>                         Float vector. Population standard deviation of each channel of the data [default: {' '.join([str(x) for x in params.std])}].
  --num_classes=<nc>                Int. Number of classes of the problem [default: {params.num_classes}].
  --gamma=<g>                       Float. Gamma of the loss if it needs it [default: {params.gamma}].
  --resize_height=<rh>              Int. New height of the input image before croping and processing [default: {params.resize_height}].
  --resize_width=<rw>               Int. New width of the input image before croping and processing [default: {params.resize_width}].
  --stride_rate=<sr>                Float. Overlap between crops of the input image in order to get more resolution full image results with better context [default: {params.stride_rate}].
  --scales=<ss>                     Float vector. Diferent zooms applied to the image in order to detect at different levels of details [default: {' '.join([str(x) for x in params.scales])}].
  --crop_height=<ch>                Int. Height of the crop that will enter the model [default: {params.crop_height}].
  --crop_width=<cw>                 Int. Width of the crop that will enter the model [default: {params.crop_width}].
  --funnel_map=<fm>                 Bool. ADE20k has some clases that are similar to water and sand, on one project it may be useful to msnuslly guide the last layer [default: {params.funnel_map}].
  --zoom_factor=<zf>                Int. Parameter to build pyConvSegNet and maybe future models [default: {params.zoom_factor}].
  --layers=<l>                      Int. Number of layers of the model [default: {params.layers}].
  --num_classes_pretrain=<ncp>      Int. Number of classes of the pretrained model (150 for ADE20k) [default: {params.num_classes_pretrain}].
  --backbone_output_stride=<bos>    Int. Parameter to build pyConvSegNet and maybe future models [default: {params.backbone_output_stride}].
  --backbone_net=<bn>               Str. Parameter to build pyConvSegNet and maybe future models [default: {params.backbone_net}].
  --pretrained_back_path=<bbp>      Str. Path to the trained parameters of the first part of a model [default: {params.pretrained_back_path}].
  --adapt_state_dict=<asd>          Bool. If the weights state dict is not compatible with the first part of the used model, apply an adaptation function manually made [default: {params.adapt_state_dict}].
  --pretrained_path=<pp>            Str. Path to the weigths of the exact full model described in the parameters [default: {params.pretrained_path}].
  
"""


def parse_args(argv):

    if len(argv[1:]) == 0:
        # This will end the script
        opts = docopt(DOCTEXT, argv=['-h'], help=True, version=None, options_first=False)
    else:
        opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)
    
    if opts['default']:
        return None, None

    experiment_metadata = SimpleNamespace(
        dataset         = opts['--dataset'],
        loss_name       = opts['--loss_name'],
        model_name      = opts['--model_name'],
        list_path       = opts['--list_path'],
        output_root     = opts['--output_root'],
        segmentation_prefix = opts['--segmentation_prefix'],
        error_prefix        = opts['--error_prefix'],
        overlapped_prefix   = opts['--overlapped_prefix']
    )

    assert experiment_metadata.segmentation_prefix != experiment_metadata.error_prefix
    assert experiment_metadata.segmentation_prefix != experiment_metadata.overlapped_prefix
    assert experiment_metadata.error_prefix != experiment_metadata.overlapped_prefix

    params = SimpleNamespace(
        # Parameters updated from data
        mean = [float(x) for x in opts['--mean']],
        std = [float(x) for x in opts['--std']],
        # Modifiable hyperparams:
        # (add new hyperparameters for any optim, loss, etc)
        num_classes = int(opts['--num_classes']),
        gamma = int(opts['--gamma']), # if focal_loss
        stride_rate = float(opts['--stride_rate']),
        scales = [float(x) for x in opts['--scales']],
        # Data Augmentation parameters:
        resize_height = int(opts['--resize_height']),
        resize_width = int(opts['--resize_width']),
        crop_height = int(opts['--crop_height']),
        crop_width = int(opts['--crop_width']),
        # PyConvSegNet hyperparams:
        funnel_map = opts['--funnel_map'] in ['True', True, 'true'],
        zoom_factor = int(opts['--zoom_factor']),
        layers = int(opts['--layers']),
        num_classes_pretrain = int(opts['--num_classes_pretrain']),
        backbone_output_stride = int(opts['--backbone_output_stride']),
        backbone_net = opts['--backbone_net'],
        # Pre-trained pathes
        pretrained_back_path = opts['--pretrained_back_path'] if opts['--pretrained_back_path'] not in ["False", "false", False, "None", "none", None] else None,
        adapt_state_dict = opts['--adapt_state_dict'] in ['True', True, 'true'],
        pretrained_path = opts['--pretrained_path'] if opts['--pretrained_path'] not in ["False", "false", False, "None", "none", None] else None
    )

    assert all(params.std) # Assert that there is no attempt to divide by 0
    assert all(params.scales) # Assert that the image at least has 1 px at processing time

    return experiment_metadata, params
