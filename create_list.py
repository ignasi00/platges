
import json
import pathlib
import sys

from docopts.help_create_list import parse_args
from preparation.lists.create_BCN_GPS_overlap_list import EXIF_homography_list
from preparation.lists.create_argusNL_list import ArgusNL_list
from preparation.lists.create_BCN_segmentation_list import platgesbcn_list
from preparation.lists.utils.split_list import split_list, split_pandas


IMAGES = 'images'
SEGMENTS = 'segments'
CLASSES = 'classes'


def save_list_GPS(save_path, list_of_lists):
    with open(save_path, 'w') as f: 
        json.dump(list_of_lists, f)

def save_list_seg(save_path, list_of_items):
    list_of_items.to_csv(save_path, header=False, index=False, columns=[IMAGES, SEGMENTS, CLASSES])

if __name__ == "__main__":
    args = parse_args(sys.argv)
    (type_, data_root, outputs_root, probs, names, img_ext, seg_ext, cls_ext, th_time, longitude, latitude, min_imgs, verbose) = args

    pathlib.Path(outputs_root).mkdir(parents=True, exist_ok=True)

    if type_ == "argusNL":
        # python3 create_list.py argusNL --data_root='/mnt/c/Users/Ignasi/Downloads/ArgusNL' --probs=1 --name=argusNL_K1-5.csv --probs=1 --name=argusNL_K2-5.csv --probs=1 --name=argusNL_K3-5.csv --probs=1 --name=argusNL_K4-5.csv --probs=1 --name=argusNL_K5-5.csv
        list_of_lists = ArgusNL_list(data_root, save_path=None, img_ext=img_ext, seg_ext=seg_ext, cls_ext=cls_ext)

        lists = split_pandas(list_of_lists, probs=probs)
        for l, n in zip(lists, names):
            save_list_seg(f'{outputs_root}{n}', l)

    elif type_ == "BCNseg":
        # python3 create_list.py BCNseg --data_root='/mnt/c/Users/Ignasi/Downloads/platgesbcn2021/' --probs=1 --names=platgesbcn2021_K1-5.csv --probs=1 --names=platgesbcn2021_K2-5.csv --probs=1 --names=platgesbcn2021_K3-5.csv --probs=1 --names=platgesbcn2021_K4-5.csv --probs=1 --names=platgesbcn2021_K5-5.csv
        list_of_lists = platgesbcn_list(data_root, save_path=None, img_ext=img_ext, seg_ext=seg_ext, cls_ext=cls_ext)

        lists = split_pandas(list_of_lists, probs=probs)
        for l, n in zip(lists, names):
            save_list_seg(f'{outputs_root}{n}', l)

    elif type_ == "BCN_GPS":
        list_of_lists = EXIF_homography_list(   data_root, save_path=None,
                                                th_time=th_time, longitude_bin_size=longitude, latitude_bin_size=latitude, min_per_bin=min_imgs,
                                                verbose=verbose)
        
        lists = split_list(list_of_lists, probs=probs)
        for l, n in zip(lists, names):
            save_list_GPS(f'{outputs_root}{n}', l)
