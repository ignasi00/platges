
from docopt import docopt


POINT_FINDER_NAME_OPTIONS = ['SIFT', "SUPERPOINTS"]

LIST_PATH = None
DATA_ROOT = None
OUTPUTS_ROOT = None

DOWNSAMPLE = 4
POINT_FINDER_NAME = 'SIFT'
#WEIGHTS_PATH = None
NMS_DIST = 4
CONF_TH = 0.015
NN_TH = 0.7
RADI = 2
RANSAC_TH = 5.0
RANSAC_MAXITERS = 1000

VERBOSE_MAIN = False
VERBOSE_TQDM = False
VERBOSE_MAT = False
VERBOSE_PLOT = False


DOCTEXT = f"""
Usage:
  panoramica_platges_GPS.py [--list_path=<lp>] [--data_root=<dr>] [--outputs_root=<or>] [--downsample=<d>] [--point_finder=<pf>] [--weights_path=<wp>] [--nms_dist=<nd>] [--conf_th=<ct>] [--nn_th=<nt>] [--radius=<r>] [--ransac_th=<rt>] [--ransac_maxiters=<rm>] [--verbose_main=<v>] [--verbose_tqdm=<vt>] [--verbose_matrix=<vm>] [--verbose_plot=<vp>]
  panoramica_platges_GPS.py -h | --help

Options:
  -h --help                             Show this screen.
  --list_path=<lp>                      Str. List used to make the dataset.
  --data_root=<dr>                      Str. Root from where the List start to search images.
  --outputs_root=<or>                    Str. Root folder where the output is stored.
  --downsample=<d>                      Int. Downsampling factor [default: {DOWNSAMPLE}].
  --point_finder=<pf>                   Str. Name of the available point finders (| {' | '.join([str(x) for x in POINT_FINDER_NAME_OPTIONS])[:-1]}) [default: {POINT_FINDER_NAME}].
  --weights_path=<wp>                   Str. Path to the file where the point_finder parameters are stored if the point_finder allows them.
  --nms_dist=<nd>                       Int. Non-Maximum Supression distance of the point_finder if it requieres this hyperparameter [default: {NMS_DIST}].
  --conf_th=<ct>                        Float. Confidence threshold of the point_finder if it requieres this hyperparameter [default: {CONF_TH}].
  --nn_th=<nt>                          Float. Nearest Neighbour threshold of the point_finder if it requieres this hyperparameter [default: {NN_TH}].
  --radius=<r>                          Float. Hyperparameter that control the radius of the found points if the point_finder allows it [default: {RADI}].
  --ransac_th=<rt>                      Float. RanSaC threshold for homography matrix estimation [default: {RANSAC_TH}].
  --ransac_maxiters=<rm>                Int. RanSaC maximum number of iterations [default: {RANSAC_MAXITERS}].
  --verbose_main=<v>                    Bool. Print the name of the currently processed images [default: {VERBOSE_MAIN}]. 
  --verbose_tqdm=<vt>                   Bool. For the current processed images, show a progress bar [default: {VERBOSE_TQDM}].
  --verbose_matrix=<vm>                 Bool. For the current processed images, print the number of RanSaC insiders 'confusion' matrix [default: {VERBOSE_MAT}].
  --verbose_plot=<vp>                   Bool. Blocking verbose. Plot the found correspondence points for each pair of images that are going to be stiched [default: {VERBOSE_PLOT}].
"""


def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    list_path = opts['--list_path']
    data_root = opts['--data_root']
    outputs_root = opts['--outputs_root']

    downsample = int(opts['--downsample'])
    point_finder = opts['--point_finder']
    weights_path = opts['--weights_path']
    nms_dist = int(opts['--nms_dist'])
    conf_th = float(opts['--conf_th'])
    nn_th = float(opts['--nn_th'])
    radi = float(opts['--radius'])
    ransac_th = float(opts['--ransac_th'])
    ransac_maxIters = int(opts['--ransac_maxiters'])

    verbose_main = True if opts['--verbose_main'] == "True" else False
    verbose_tqdm = opts['--verbose_tqdm'] == "True"
    verbose_mat = opts['--verbose_matrix'] == "True"
    verbose_plot = opts['--verbose_plot'] == "True"

    if point_finder not in POINT_FINDER_NAME_OPTIONS:
        print(f"Point Finder {point_finder} not available yet. See available options below:", end="\n")
        print(DOCTEXT)
        raise Exception(f"Point Finder {point_finder} not available yet.")

    args = (list_path, data_root, outputs_root, downsample, point_finder, weights_path, nms_dist, conf_th, nn_th, radi, ransac_th, ransac_maxIters, verbose_main, verbose_tqdm, verbose_mat, verbose_plot)
    return args

