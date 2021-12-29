
from docopt import docopt


DATA_PATH = '/mnt/c/Users/Ignasi/Downloads/4fotos' #'/mnt/c/Users/Ignasi/Downloads/PlatgesBCN/dron'
SAVE_PATH = '/mnt/c/Users/Ignasi/Downloads/panorama_test.jpg'
DOWNSAMPLE = 4
CLUSTER = True
TH_TIME=60
LONGITUDE_BIN_SIZE=180
LATITUDE_BIN_SIZE=180
MIN_PER_BIN=1

POINT_FINDER_NUMBER = 1
RANSAC_TH = 5

WEIGHTS_PATH='extern/SuperPointPretrainedNetwork/superpoint_v1.pth'
NMS_DIST=4
CONF_THRESH=0.015
NN_THRESH=0.7
RADI = 2


# There is debugging modes on mode = 2 and 3; if there where not, mode should morf into arguments and reduce the available options on SIFT mode
DOCTEXT = f"""
Usage:
  platges_panoramica [--no_cluster|[--time=<t>] [--longitude_bin=<lo>] [--latitude_bin=<la>] [--min_imgs_bin=<mb>]] [--data_path=<dp>] [--save_dir=<sd>] [--downsample=<ds>] [--mode=<m>] [--superpoints_weigths_path=<swp>] [--nms_dist=<nd>] [--conf_th=<ct>] [--nn_th=<nt>] [--radius=<r>] [--ransac_th=<rt>] [--cuda] [--verbose] [--plot]
  platges_panoramica -h | --help

Options:
  -h --help                             Show this screen.
  --no_cluster                          All images on the selected folder will be used together
  --cluster                             EXIF metadata will be used to group images
  --time=<t>                            Float. Time span of each group in seconds [default: {TH_TIME}].
  --longitude_bin=<lo>                  Float. Longitude span of each group in seconds [default: {LONGITUDE_BIN_SIZE}].
  --latitude_bin=<la>                   Float. Latitude span of each group in seconds [default: {LATITUDE_BIN_SIZE}].
  --min_imgs_bin=<mb>                   Int. Minimum number of images on each group; if a group has not enough, its images will be merged to the nearest groups [default: {MIN_PER_BIN}].
  --data_path=<dp>                      Set the folder with the source images [default: {DATA_PATH}].
  --save_dir=<sd>                       Set the folder where the results will be stored [default: {SAVE_PATH}].
  --downsample=<ds>                     Int. Set the downsample factor [default: {DOWNSAMPLE}].
  --mode=<m>                            Int. Algorithm type to find and describe keypoints (0: SIFT, 1: SUPERPOINTS) [default: {POINT_FINDER_NUMBER}].
  --superpoints_weigths_path=<swp>      Path to the file where superpoints weights are saved [default: {WEIGHTS_PATH}].
  --nms_dist=<nd>                       Int. Superpoints non-maximum supression distance [default: {NMS_DIST}].
  --conf_th=<ct>                        Float. Superpoints confidance threshold [default: {CONF_THRESH}].
  --nn_th=<nt>                          Float. Superpoints nearest neighbours threshold [default: {NN_THRESH}]
  --radius=<r>                          Float. Superpoints multiplicative control over the influence point found [default: {RADI}].
  --ransac_th=<rt>                      Float. RanSaC threshold for homography matrix estimation [default: {RANSAC_TH}].
  --cuda                                If set, cuda will be used.
  --verbose                             If set, there will be text information about the progress.
  --plot                                If set, for each pair of images, the program will pause and plot the matched points.
"""

def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv, help=True, version=None, options_first=False)

    cluster = not opts['--no_cluster']
    time = float(opts['--time'])
    longitude_bin_size = float(opts['--longitude_bin'])
    latitude_bin_size = float(opts['--latitude_bin'])
    min_imgs_bin = int(opts['--min_imgs_bin'])
    data_path = opts['--data_path']
    save_dir = opts['--save_dir']
    downsample = int(opts['--downsample'])
    point_finder_number = int(opts['--mode'])
    weights_path = opts['--superpoints_weigths_path']
    nms_dist = int(opts['--nms_dist'])
    conf_th = float(opts['--conf_th'])
    nn_th = float(opts['--nn_th'])
    radi = float(opts['--radius'])
    ransac_th = float(opts['--ransac_th'])
    cuda = opts['--cuda']
    verbose = opts['--verbose']
    plot = opts['--plot']

    args = (cluster, time, longitude_bin_size, latitude_bin_size, min_imgs_bin, data_path, save_dir, downsample, point_finder_number, weights_path, nms_dist, conf_th, nn_th, radi, ransac_th, cuda, verbose, plot)
    return args
