
from docopt import docopt
from types import SimpleNamespace


POINT_FINDER_NAME_OPTIONS = ['SIFT', "SUPERPOINTS"]

experiment_params = SimpleNamespace(
    list_path = './data_lists/platges2021_all.json',
    data_root = '', #'/mnt/c/Users/Ignasi/Downloads/PlatgesBCN/dron/'
    outputs_root = './outputs/panoramica_platges_GPS/',
    downsample = 4,
    point_finder_name = 'SIFT',
    # HANDPICKED: it is implemented but not available because it requieres some hard-wireing (better config files).
    # HANDPICKED: The best solution would be implement a handpicked model and implement the hand picked as a dataset (with its data_list).
    handpicked = None,
    weights_path = 'extern/SuperPointPretrainedNetwork/superpoint_v1.pth',
    nms_dist = 4,
    conf_th = 0.015,
    nn_th = 0.7,
    radi = 2,
    ransac_th = 5.0,
    ransac_maxIters = 1000,
    num_imgs = None
)

VERBOSE_MAIN = False
VERBOSE_TQDM = True
VERBOSE_MAT = False
VERBOSE_PLOT = False


DOCTEXT = f"""
Usage:
  panoramica_platges_GPS.py [--list_path=<lp>] [--data_root=<dr>] [--outputs_root=<or>] [--downsample=<d>] [--point_finder=<pf>] [--weights_path=<wp>] [--nms_dist=<nd>] [--conf_th=<ct>] [--nn_th=<nt>] [--radius=<r>] [--ransac_th=<rt>] [--ransac_maxiters=<rm>] [--verbose_main=<v>] [--verbose_tqdm=<vt>] [--verbose_matrix=<vm>] [--verbose_plot=<vp>]
  panoramica_platges_GPS.py -h | --help

Options:
  -h --help                             Show this screen.
  --list_path=<lp>                      Str. List used to make the dataset [default: {experiment_params.list_path}].
  --data_root=<dr>                      Str. Root from where the List start to search images [default: {experiment_params.data_root}].
  --outputs_root=<or>                   Str. Root folder where the output is stored [default: {experiment_params.outputs_root}].
  --downsample=<d>                      Int. Downsampling factor [default: {experiment_params.downsample}].
  --point_finder=<pf>                   Str. Name of the available point finders (| {' | '.join([str(x) for x in POINT_FINDER_NAME_OPTIONS])}) [default: {experiment_params.point_finder_name}].
  --weights_path=<wp>                   Str. Path to the file where the point_finder parameters are stored if the point_finder allows them [default: {experiment_params.weights_path}].
  --nms_dist=<nd>                       Int. Non-Maximum Supression distance of the point_finder if it requieres this hyperparameter [default: {experiment_params.nms_dist}].
  --conf_th=<ct>                        Float. Confidence threshold of the point_finder if it requieres this hyperparameter [default: {experiment_params.conf_th}].
  --nn_th=<nt>                          Float. Nearest Neighbour threshold of the point_finder if it requieres this hyperparameter [default: {experiment_params.nn_th}].
  --radius=<r>                          Float. Hyperparameter that control the radius of the found points if the point_finder allows it [default: {experiment_params.radi}].
  --ransac_th=<rt>                      Float. RanSaC threshold for homography matrix estimation [default: {experiment_params.ransac_th}].
  --ransac_maxiters=<rm>                Int. RanSaC maximum number of iterations [default: {experiment_params.ransac_maxIters}].
  --verbose_main=<v>                    Bool. Print the name of the currently processed images [default: {VERBOSE_MAIN}]. 
  --verbose_tqdm=<vt>                   Bool. For the current processed images, show a progress bar [default: {VERBOSE_TQDM}].
  --verbose_matrix=<vm>                 Bool. For the current processed images, print the number of RanSaC insiders 'confusion' matrix [default: {VERBOSE_MAT}].
  --verbose_plot=<vp>                   Bool. Blocking verbose. Plot the found correspondence points for each pair of images that are going to be stiched [default: {VERBOSE_PLOT}].
"""


def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    experiment_params = SimpleNamespace(
        list_path = opts['--list_path'],
        data_root = opts['--data_root'],
        outputs_root = opts['--outputs_root'],
        downsample          = int(opts['--downsample']),
        point_finder_name   = opts['--point_finder'],
        # HANDPICKED: it is implemented but not available because it requieres some hard-wireing (better config files).
        # HANDPICKED: The best solution would be implement a handpicked model and implement the hand picked as a dataset (with its data_list).
        handpicked = None,
        weights_path        = opts['--weights_path'],
        nms_dist            = int(opts['--nms_dist']),
        conf_th             = float(opts['--conf_th']),
        nn_th               = float(opts['--nn_th']),
        radi                = float(opts['--radius']),
        ransac_th           = float(opts['--ransac_th']),
        ransac_maxIters     = int(opts['--ransac_maxiters']),
        num_imgs            = None
    )

    verbose_main = True if opts['--verbose_main'] == "True" else False
    verbose_tqdm = opts['--verbose_tqdm'] == "True"
    verbose_mat = opts['--verbose_matrix'] == "True"
    verbose_plot = opts['--verbose_plot'] == "True"

    if experiment_params.point_finder_name not in POINT_FINDER_NAME_OPTIONS:
        print(f"Point Finder {experiment_params.point_finder_name} not available yet. See available options below:", end="\n\n")
        docopt(DOCTEXT, argv=['-h'], help=True, version=None, options_first=False)

    args = (experiment_params, verbose_main, verbose_tqdm, verbose_mat, verbose_plot)
    return args

