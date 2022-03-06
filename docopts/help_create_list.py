
from docopt import docopt


#DATA_ROOT = None
OUTPUT_ROOT = './data_lists/'

IMG_EXT = '.jpg'
SEG_EXT = '.segments.pkl'
CLS_EXT = '.classes.pkl'

TH_TIME = 60
LONGITUDE_BIN_SIZE = 0.05
LATITUDE_BIN_SIZE = 0.05
MIN_PER_BIN = 1

#PROBS = None
#NAMES = None

VERBOSE = False


DOCTEXT = f"""
Usage:
  create_list.py (argusNL | BCNseg) --data_root=<dr> (--probs=<p>... --names=<n>...) [--outputs_root=<or>] [--img_ext=<ie>] [--seg_ext=<se>] [--cls_ext=<ce>] [--verbose=<v>]
  create_list.py BCN_GPS --data_root=<dr> (--probs=<p>... --names=<n>...) [--outputs_root=<or>] [--th_time=<tt>] [--longitude=<lo>] [--latitude=<la>] [--min_imgs=<mi>] [--verbose=<v>]
  create_list.py -h | --help

Options:
  -h --help                             Show this screen.
  --data_root=<dr>                      Str. Path of the folder where the data is stored.
  --probs=<p>                           Float vector. Positive numbers that represent a fraction of the data available, they will be normalized (divided by their sum).
  --names=<n>                           Str vector. Name associated to each probs value, also the list output name.
  --outputs_root=<or>                    Str. Path where the list will be generated [default: {OUTPUT_ROOT}].
  --img_ext=<ie>                        Str. Image files extension for the lists that requiere it [default: {IMG_EXT}].
  --seg_ext=<se>                        Str. Segment files extension for the lists that requiere it [default: {SEG_EXT}].
  --cls_ext=<ce>                        Str. Clase files extension for the lists that requiere it [default: {CLS_EXT}].
  --th_time=<tt>                        Float. Size of the time bin of a histogram for the BCN_GPS datasets (which clusters images) [default: {TH_TIME}].
  --longitude=<lo>                      Float. Size of the longitude bin of a histogram for the BCN_GPS datasets (which clusters images) [default: {LONGITUDE_BIN_SIZE}].
  --latitude=<la>                       Float. Size of the latitude bin of a histogram for the BCN_GPS datasets (which clusters images) [default: {LATITUDE_BIN_SIZE}].
  --min_imgs=<mi>                       Int. Minimum number of images per cluster of the BCN_GPS, if not reached, images will be added to its nearest bin [default: {MIN_PER_BIN}].
  --verbose=<v>                         Bool. Some list creation script may show information as it runs, this parameter manage it [default: {VERBOSE}].
"""


def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    type_ = None
    if opts['argusNL'] : type_ = "argusNL"
    elif opts['BCNseg'] : type_ = "BCNseg"
    elif opts['BCN_GPS'] : type_ = "BCN_GPS"

    data_root = opts['--data_root']
    outputs_root = opts['--outputs_root']
    probs = [float(x) for x in opts['--probs']]
    probs = [x/sum(probs) for x in probs]
    names = opts['--names']

    img_ext = opts['--img_ext']
    seg_ext = opts['--seg_ext']
    cls_ext = opts['--cls_ext']

    th_time = float(opts['--th_time'])
    longitude = float(opts['--longitude'])
    latitude = float(opts['--latitude'])
    min_imgs = int(opts['--min_imgs'])

    verbose = opts['--verbose'] == "True"

    args = (type_, data_root, outputs_root, probs, names, img_ext, seg_ext, cls_ext, th_time, longitude, latitude, min_imgs, verbose)
    return args

