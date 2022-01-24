
from datetime import datetime
from docopt import docopt


DATA_PATH = '/home/ignasi/platges/data_lists/argusNL_all.csv'
TRAIN_PROB = 0.7
KFOLDS_K = 5
RESIZE_HEIGHT = int(1024 / 2)
RESIZE_WIDTH = int(1392 / 2)
CROP_HEIGHT = 473
CROP_WIDTH = 473
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
VALUE_SCALE = 255
BATCH_SIZE = 2 #9
LAYERS = 152
NUM_CLASSES = 3
ZOOM_FACTOR = 8
BACKBONE_OUTPUT_STRIDE = 8
BACKBONE_NET = "pyconvresnet"
PRETRAINED_PATH = '/home/ignasi/platges/extern/pyconvsegnet/ade20k_trainval120epochspyconvresnet152_pyconvsegnet.pth'
LEARNING_RATE = 0.0001

EXPERIMENT_NAME = datetime.now().strftime("%Y%m%d_testsystem")
ENTITY = "ignasi00"
LOG_FREQ = int(TRAIN_PROB * 188 / BATCH_SIZE)
NUM_EPOCHS = 30
VAL_EPOCH_FREQ = 2
EARLY_STOP_MEMORY = 3
#FILENAME = "./model_parameters/testsystem_model.onnx"
FILENAME = "./model_parameters/testsystem_model.pth"

CUDA = False


DOCTEXT = f"""
Usage:
  segmentation_train argus pyconvsegnet [--data_path=<dp>] [--train_prob=<tp>] [--resize_h=<rh>] [--resize_w=<rw>] [--crop_h=<ch>] [--crop_w=<cw>] [--mean=<m>...] [--std=<s>...] [--value_scale=<vs>] [--batch_size=<bs>] [--layers=<l>] [--num_classes=<nc>] [--zoom_factor=<zf>] [--backbone_output_stride=<bos>] [--backbone_net=<bn>] [--pretrained_path=<pp>] [--learning_rate=<lr>] [--experiment_name=<en>] [--entity=<e>] [--log_freq=<lf>] [--num_epochs=<ne>] [--val_epoch_freq=<vef>] [--early_stop_memory=<esm>] [--output_filename=<of>] [--cuda]
  segmentation_train argus pyconvsegnet kfolds [--data_path=<dp>] [--kfolds_k=<kk>] [--resize_h=<rh>] [--resize_w=<rw>] [--crop_h=<ch>] [--crop_w=<cw>] [--mean=<m>...] [--std=<s>...] [--value_scale=<vs>] [--batch_size=<bs>] [--layers=<l>] [--num_classes=<nc>] [--zoom_factor=<zf>] [--backbone_output_stride=<bos>] [--backbone_net=<bn>] [--pretrained_path=<pp>] [--learning_rate=<lr>] [--experiment_name=<en>] [--entity=<e>] [--log_freq=<lf>] [--num_epochs=<ne>] [--val_epoch_freq=<vef>] [--early_stop_memory=<esm>] [--output_filename=<of>] [--cuda]
  segmentation_train -h | --help


#####################################
segmentation_train <used dataset> <used model> [options]

<used dataset> is a code name for the dataset or datasets that will be used for the test.
<used model> is a code name for the model name.
<initial output model> is a code name for the direct output of the model; when the output is not directly the our problem model, it will be somehow adapted

vector options like [--mean=<m>...] or [--std=<s>...] have to be inputed like: --mean=3 --mean=6 --mean=10 for a [3, 6, 10] output
#####################################


Options:
  -h --help                             Show this screen.
  --data_path=<dp>                      Set the folder with the source images [default: {DATA_PATH}].
  --train_prob=<tp>                     Float. Percentage of dataset used for validation [default: {TRAIN_PROB}].
  --kfolds_k=<kk>                       Int. When using kfolds, the number of folds [default: {KFOLDS_K}].
  --resize_h=<rh>                       Int. The height to resize the input image to [default: {RESIZE_HEIGHT}].
  --resize_w=<rw>                       Int. The width to resize the input image to [default: {RESIZE_WIDTH}].
  --crop_h=<ch>                         Int. The height of the crops the network will process [default: {CROP_HEIGHT}].
  --crop_w=<cw>                         Int. The width of the crops the network will process [default: {CROP_WIDTH}].
  --mean=<m>                            Float vector. PyConvSegNet means [default: {' '.join([str(x) for x in MEAN])}].
  --std=<s>                             Float vector. PyConvSegNet standar deviations [default: {' '.join([str(x) for x in STD])}].
  --value_scale=<vs>                    Float. PyConvSegNet scale [default: {VALUE_SCALE}].
  --batch_size=<bs>                     Int. Number of images simultaneously processed [default: {BATCH_SIZE}].
  --layers=<l>                          Int. Number of network layers [default: {LAYERS}].
  --num_classes=<nc>                    Int. Number of output classes the model have [default: {NUM_CLASSES}].
  --zoom_factor=<zf>                    Int. PyConvSegNet zoom factor [default: {ZOOM_FACTOR}].
  --backbone_output_stride=<bos>        Int. PyConvSegNet backbone output strice [default: {BACKBONE_OUTPUT_STRIDE}].
  --backbone_net=<bn>                   Backbone of the model [default: {BACKBONE_NET}].
  --pretrained_path=<pp>                Path to the file where the models weights are saved [default: {PRETRAINED_PATH}].
  --learning_rate=<lr>                  Float. Value that determine how much the gradients affect the parameters [default: {LEARNING_RATE}].
  --experiment_name=<en>                Name that identify the current experminet on the wandb [defaulf: {EXPERIMENT_NAME}].
  --entity=<e>                          Name that identify how is updating the wandb [default: {ENTITY}].
  --log_freq=<lf>                       Int. The number of optimization steps before logging gradients and parameters [default: {LOG_FREQ}].
  --num_epochs=<ne>                     Int. The maximum number of training epochs [default: {NUM_EPOCHS}].
  --val_epoch_freq=<vef>                Int. The number of training epochs before a validation epoch [default: {VAL_EPOCH_FREQ}].
  --early_stop_memory=<esm>             Int. The number of consecutive validation epochs with worse loss in order to early stop [default: {EARLY_STOP_MEMORY}].
  --output_filename=<of>                Name of the file where the model weights will be stored [default: {FILENAME}].
  --cuda                                If set, cuda will be used.
"""

def process_mode(opts):
  if opts['argus'] and opts['pyconvsegnet'] and not opts['kfolds']: return "argus_pyConvSegNet"
  elif opts['argus'] and opts['pyconvsegnet'] and opts['kfolds']: return "argus_pyConvSegNet_kfolds"

def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv, help=True, version=None, options_first=False)

    data_path = opts['--data_path']
    #
    batch_size = int(opts['--batch_size'])
    #
    learning_rate = float(opts['--learning_rate'])
    experiment_name = opts['--experiment_name']
    entity = opts['--entity']
    log_freq = int(opts['--log_freq'])
    num_epochs = int(opts['--num_epochs'])
    val_epoch_freq = int(opts['--val_epoch_freq'])
    early_stop_memory = int(opts['--early_stop_memory'])
    filename = opts['--output_filename']
    cuda = opts['--cuda']

    mode = process_mode(opts)

    args = []
    if mode == "argus_pyConvSegNet" or mode == "argus_pyConvSegNet_kfolds":
        #
        resize_height = int(opts['--resize_h'])
        resize_width = int(opts['--resize_w'])
        crop_height = int(opts['--crop_h'])
        crop_width = int(opts['--crop_w'])
        mean = [float(x) for x in opts['--mean']]
        std = [float(x) for x in opts['--std']]
        value_scale = float(opts['--value_scale'])
        #
        layers = int(opts['--layers'])
        num_classes = int(opts['--num_classes'])
        zoom_factor = int(opts['--zoom_factor'])
        backbone_output_stride = int(opts['--backbone_output_stride'])
        backbone_net = opts['--backbone_net']
        pretrained_path = opts['--pretrained_path']

        if mode == "argus_pyConvSegNet":
          train_prob = float(opts['--train_prob'])
          args = (data_path, train_prob, resize_height, 
              resize_width, crop_height, crop_width, 
              mean, std, value_scale, batch_size,
              layers, num_classes, zoom_factor, 
              backbone_output_stride, backbone_net, 
              pretrained_path, learning_rate, 
              experiment_name, entity, log_freq, num_epochs, 
              val_epoch_freq, early_stop_memory, filename, cuda)
        else:
          kfolds_k = int(opts['--kfolds_k'])
          args = (data_path, kfolds_k, resize_height, 
              resize_width, crop_height, crop_width, 
              mean, std, value_scale, batch_size,
              layers, num_classes, zoom_factor, 
              backbone_output_stride, backbone_net, 
              pretrained_path, learning_rate, 
              experiment_name, entity, log_freq, num_epochs, 
              val_epoch_freq, early_stop_memory, filename, cuda)
    
    return args, mode