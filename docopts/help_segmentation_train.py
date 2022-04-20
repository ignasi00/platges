
from datetime import datetime
from docopt import docopt
from types import SimpleNamespace


experiment_metadata = SimpleNamespace(
    experiment_type = "segments_argusNL",
    dataset         = 'argusNL',
    training_type   = 'kfolds',
    loss_name       = 'dice_loss',
    optim_name      = 'Adam',
    model_name      = 'pyConvSegNet',
    lists_paths     = [f"./data_lists/argusNL_K{i}-5.csv" for i in range(1, 5+1)],
    num_val_folds   = 1,
    num_folds       = 1,
    output_root     = f"./outputs/segmentation/",
    models_root     = f"model_parameters/segmentation/",
    experiment_name = datetime.now().strftime(f"%Y%m%d%H%M%S_segmentation"),
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
    pretrained_back_path = './model_parameters/original_pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth',
    adapt_state_dict = True,
    pretrained_path = None
)

DOCTEXT = f"""
Usage:
  segmentation_train.py [options] [--lists_paths=<lp>...] [--betas=<b>...]
  segmentation_train.py default
  segmentation_train.py -h | --help

No args, -h or --help will show this screen.
Mode default will use the values on segmentation_train.py SimpleNamespaces.
Setting at least 1 value will use the setted values and the defaults from Docopts.

[--lists_paths=<lp>...] and similar parameters are assigned as --lists_paths='bla/bla/bla.bla' --lists_paths='bla/bla/bla.bla' --lists_paths='bla/bla/bla.bla'

Options:
  -h --help                         Show this screen.

  --experiment_type=<et>            Str. Name of the current experiment (local folders name usage) [default: {experiment_metadata.experiment_type}].
  --dataset=<d>                     Str. Name of the dataset in use (argusNL, platgesBCN, etc) [default: {experiment_metadata.dataset}].
  --training_type=<tt>              Str. Name of the training type in use (vanilla, kfolds, etc) [default: {experiment_metadata.training_type}].
  --loss_name=<ln>                  Str. Name of the loss type to use (dice_loss, focal_loss, etc) [default: {experiment_metadata.loss_name}].
  --optim_name=<on>                 Str. Name of the optimizer type to use (Adam, etc) [default: {experiment_metadata.optim_name}].
  --model_name=<mn>                 Str. Name of the model type to use (pyConvSegNet, etc) [default: {experiment_metadata.model_name}].
  --lists_paths=<lp>                Str vector. Pathes to the lists to be used [default: {' '.join([str(x) for x in experiment_metadata.lists_paths])}].
  --num_val_folds=<nvf>             Int. In case of kfolds training, how many lists of the lists_paths vector are used for validation [default: {experiment_metadata.num_val_folds}].
  --output_root=<or>                Str. Path to the folder where the outputs will be stored [default: {experiment_metadata.output_root}].
  --models_root=<mr>                Str. Path to the folder where the trained parameters will be stored [default: {experiment_metadata.models_root}].
  --experiment_name=<en>            Str. Name of the file of trained parameters (extension .pt) [default: {experiment_metadata.experiment_name}].
  --initial_epoch=<ie>              Int. Offset on the WandB epoch step logging [default: {experiment_metadata.initial_epoch}].
  --project_name=<pn>               Str. WandB project name [default: {experiment_metadata.project_name}].
  --entity=<e>                      Str. WandB user name [default: {experiment_metadata.entity}].

  --num_classes=<nc>                Int. Number of classes of the problem [default: {params.num_classes}].
  --num_epochs=<ne>                 Int. Number of epochs of the training [default: {params.num_epochs}].
  --learning_rate=<lr>              Float. Learning rate of the optimizer if it needs it (TODO: more complex learing rates) [default: {params.learning_rate}].
  --batch_size=<bs>                 Int. Number of inputs to be processed before doing a optimization step [default: {params.batch_size}].
  --gamma=<g>                       Float. Gamma of the loss if it needs it [default: {params.gamma}].
  --betas=<b>                       Float vector. Betas of the optimizer if it needs them [default: {' '.join([str(x) for x in params.betas])}].
  --weight_decay=<wd>               Float. Weights decay from the optimizer if it needs it [default: {params.weight_decay}].
  --resize_height=<rh>              Int. New height of the input image before croping and processing [default: {params.resize_height}].
  --resize_width=<rw>               Int. New width of the input image before croping and processing [default: {params.resize_width}].
  --crop_height=<ch>                Int. Height of the crop that will enter the model [default: {params.crop_height}].
  --crop_width=<cw>                 Int. Width of the crop that will enter the model [default: {params.crop_width}].
  --scale_limit=<scl>               Float. Maximum percentage of scale variation (+-) of the input image on the data augmentation step [default: {params.scale_limit}].
  --shift_limit=<shl>               Float. Maximum percentage of shift variation (+-) of the input image on the data augmentation step [default: {params.shift_limit}].
  --rotate_limit=<rl>               Float. Maximum percentage of rotation variation (+-) of the input image on the data augmentation step [default: {params.rotate_limit}].
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
        experiment_type = opts['--experiment_type'],
        dataset         = opts['--dataset'],
        training_type   = opts['--training_type'],
        loss_name       = opts['--loss_name'],
        optim_name      = opts['--optim_name'],
        model_name      = opts['--model_name'],
        lists_paths     = opts['--lists_paths'],
        num_val_folds   = int(opts['--num_val_folds']),
        num_folds       = 1,
        output_root     = opts['--output_root'],
        models_root     = opts['--model_root'],
        experiment_name = opts['--experiment_name'],
        initial_epoch   = int(opts['--initial_epoch']),
        project_name    = opts['--project_name'],
        entity          = opts['--entity']
    )

    params = SimpleNamespace(
        # Parameters updated from data
        mean = 0,
        std = 1,
        # Modifiable hyperparams:
        # (add new hyperparameters for any optim, loss, etc)
        num_classes = int(opts['--num_classes']),
        num_epochs = int(opts['--num_epochs']),
        learning_rate = float(opts['--learning_rate']),
        batch_size = int(opts['--batch_size']),
        gamma = int(opts['--gamma']), # if focal_loss
        betas = tuple(float(x) for x in opts['--betas']), # if Adam optim
        weight_decay = float(opts['--weight_decay']),
        # Data Augmentation parameters:
        resize_height = int(opts['--resize_height']),
        resize_width = int(opts['--resize_width']),
        crop_height = int(opts['--crop_height']),
        crop_width = int(opts['--crop_width']),
        scale_limit = float(opts['--scale_limit']),
        shift_limit = float(opts['--shift_limit']),
        rotate_limit = float(opts['--rotate_limit']),
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

    return experiment_metadata, params
