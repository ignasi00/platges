
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
  create_list.py (argusNL | BCNseg) --data_root=<dr> (--probs=<p>... --names=<ns>...) [--outputs_root=<or>] [--img_ext=<ie>] [--seg_ext=<se>] [--cls_ext=<ce>] [--verbose=<v>]
  create_list.py BCN_GPS --data_root=<dr> (--probs=<p>... --names=<ns>...) [--outputs_root=<or>] [--th_time=<tt>] [--longitude=<lo>] [--latitude=<la>] [--min_imgs=<mi>] [--verbose=<v>]
  create_list.py correspondances --pto_filename=<pf> --name=<n> [--outputs_root=<or>] [--windows|--windows_path=<wp>]
  create_list.py -h | --help

Options:
  -h --help                             Show this screen.
  --windows                             If set, the backslashes will become / and the root will be set at /mnt/c/.
  --windows_path=<wp>                   Str. the backslashes will become / and the root will be set at its value.
  --data_root=<dr>                      Str. Path of the folder where the data is stored.
  --pto_filename=<pf>                   Str. Path to the pto file where the data is stored.
  --probs=<p>                           Float vector. Positive numbers that represent a fraction of the data available, they will be normalized (divided by their sum).
  --names=<ns>                          Str vector. Name associated to each probs value, also the list output name.
  --name=<n>                            Str. Name for the folder and files used to store the correspondances at the output_root, the extensions will be added automatically.
  --outputs_root=<or>                   Str. Path where the list will be generated [default: {OUTPUT_ROOT}].
  --img_ext=<ie>                        Str. Image files extension for the lists that requiere it [default: {IMG_EXT}].
  --seg_ext=<se>                        Str. Segment files extension for the lists that requiere it [default: {SEG_EXT}].
  --cls_ext=<ce>                        Str. Clase files extension for the lists that requiere it [default: {CLS_EXT}].
  --th_time=<tt>                        Float. Size of the time bin of a histogram for the BCN_GPS datasets (which clusters images) [default: {TH_TIME}].
  --longitude=<lo>                      Float. Size of the longitude bin of a histogram for the BCN_GPS datasets (which clusters images) [default: {LONGITUDE_BIN_SIZE}].
  --latitude=<la>                       Float. Size of the latitude bin of a histogram for the BCN_GPS datasets (which clusters images) [default: {LATITUDE_BIN_SIZE}].
  --min_imgs=<mi>                       Int. Minimum number of images per cluster of the BCN_GPS, if not reached, images will be added to its nearest bin [default: {MIN_PER_BIN}].
  --verbose=<v>                         Bool. Some list creation script may show information as it runs, this parameter manage it [default: {VERBOSE}].

FREQUENT ISSUES:
  BCNseg images are .JPG and not .jpg, it requieres --img_ext=.JPG

"""


def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    type_ = None
    if opts['argusNL'] : type_ = "argusNL"
    elif opts['BCNseg'] : type_ = "BCNseg"
    elif opts['BCN_GPS'] : type_ = "BCN_GPS"
    elif opts['correspondances'] : type_ = "correspondances"

    if type_ == "correspondances":
        data_root = opts['--pto_filename']
        names = [opts['--name']]
    else:
        data_root = opts['--data_root']
        names = opts['--names']
    
    windows = opts['--windows'] or opts['--windows_path'] is not None
    windows_path = opts['--windows_path']

    outputs_root = opts['--outputs_root']
    probs = [float(x) for x in opts['--probs']]
    probs = [x/sum(probs) for x in probs]

    img_ext = opts['--img_ext']
    seg_ext = opts['--seg_ext']
    cls_ext = opts['--cls_ext']

    th_time = float(opts['--th_time'])
    longitude = float(opts['--longitude'])
    latitude = float(opts['--latitude'])
    min_imgs = int(opts['--min_imgs'])

    verbose = opts['--verbose'] == "True"

    args = (type_, data_root, outputs_root, probs, names, img_ext, seg_ext, cls_ext, th_time, longitude, latitude, min_imgs, windows, windows_path, verbose)
    return args

