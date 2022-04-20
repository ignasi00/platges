
import pathlib
import sys

from docopts.help_prepare_data import parse_args
from preparation.data.prepare_numpy_from_CVAT_XML_images import main as prepare_numpy_from_CVAT_XML_images


if __name__ == "__main__":

    args = parse_args(sys.argv)
    (type_, annotations_path, outputs_root, interest_labels) = args

    pathlib.Path(outputs_root).mkdir(parents=True, exist_ok=True)

    if type_ == "numpy_from_CVAT_images":
        prepare_numpy_from_CVAT_XML_images(annotations_path, outputs_root, interest_labels)
