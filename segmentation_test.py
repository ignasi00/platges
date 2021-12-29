
import sys

from docopts.help_segmentation_test import parse_args

import argus_pyConvSegNet_test # it work although there is no __init__.py because it is in the same directory, TODO: make sure it works calling the script from other directories


if __name__ == "__main__":
    
    args, mode = parse_args(sys.argv[1:])

    if mode == "argus_pyConvSegNet":
        argus_pyConvSegNet_test.main(*args)