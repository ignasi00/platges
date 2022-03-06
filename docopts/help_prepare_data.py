
from docopt import docopt


DOCTEXT = f"""
Usage:
  prepare_data.py numpy_from_CVAT_images --annotations_path=<ap> --outputs_root=<or> [--interest_labels=<il>...]
  prepare_data.py -h | --help

Options:
  -h --help                             Show this screen.
  --annotations_path=<ap>               Str. File path to the source annotations.
  --outputs_root=<or>                   Str. Folder path where the new data will be stored.
  --interest_labels=<il>                Str vector. Names of classes that will be used.
"""



def parse_args(argv):
    
    opts = docopt(DOCTEXT, argv=argv[1:], help=True, version=None, options_first=False)

    type_ = None
    if opts['numpy_from_CVAT_images'] : type_ = "numpy_from_CVAT_images"
    
    annotations_path = opts["--annotations_path"]
    outputs_root = opts["--outputs_root"]
    interest_labels = opts["--interest_labels"]
    if len(interest_labels) == 0 : interest_labels = None
    
    args = (type_, annotations_path, outputs_root, interest_labels)
    return args
