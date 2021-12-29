
from docopt import docopt




DOCTEXT = f"""
Usage:
  segmentation_train 

Options:
  -h --help                             Show this screen.
"""

def parse_args(argv):
    opts = docopt(DOCTEXT, argv=argv, help=True, version=None, options_first=False)

    # x = int(opts['--value']) # --value=<x>    Whatever [default: 4].

    return 