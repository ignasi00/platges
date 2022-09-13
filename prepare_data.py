
import pathlib
import sys

from docopts.help_prepare_data import parse_args
from preparation.data.prepare_numpy_from_CVAT_XML_images import main as prepare_numpy_from_CVAT_XML_images


def save_image_list(save_path, list_of_lists):
    with open(save_path, 'w') as f:
        json.dump(list_of_lists, f)

def save_correspondances(save_path, correspondances_df):
    correspondances_df.to_csv(save_path, index=False)

if __name__ == "__main__":

    args = parse_args(sys.argv)
    (type_, annotations_path, outputs_root, interest_labels) = args

    pathlib.Path(outputs_root).mkdir(parents=True, exist_ok=True)

    if type_ == "numpy_from_CVAT_images":
        classes = None
        if interest_labels == None:
            classes = {"sorra"  : 2, "aigua" : 1, "sorra_gossos" : 8, "aigua_gossos" : 4}
        prepare_numpy_from_CVAT_XML_images(annotations_path, outputs_root, interest_labels, classes)
