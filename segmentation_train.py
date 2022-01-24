
import sys

from docopts.help_segmentation_train import parse_args

import argus_pyConvSegNet_train


if __name__ == "__main__":
    
    args, mode = parse_args(sys.argv[1:])

    if mode == "argus_pyConvSegNet":
        argus_pyConvSegNet_train.main(*args)