
import os
import pandas as pd

from .utils.split_list import split_pandas


IMAGES = 'images'
SEGMENTS = 'segments'
CLASSES = 'classes'


def _tabulate_files(folder_path, img_ext=None, seg_ext=None, cls_ext=None):
    img_ext = img_ext or '.JPG'
    seg_ext = seg_ext or '.segments.pkl' # TODO: maybe converting pkl to pt could speed all up; platgesbcn made compatible with argusNL
    cls_ext = cls_ext or '.classes.pkl'

    folder_path = os.path.abspath(folder_path)
    list_of_files = pd.DataFrame({'filenames' : [f.path for f in os.scandir(folder_path) if f.is_file()]})

    list_of_items = pd.DataFrame()
    list_of_items[IMAGES] = list_of_files[list_of_files['filenames'].str.endswith(img_ext, na=False)]
    list_of_items[SEGMENTS] = list_of_items[IMAGES].str[:-len(img_ext)] + seg_ext
    list_of_items[CLASSES] = list_of_items[IMAGES].str[:-len(img_ext)] + cls_ext

    list_of_files = list_of_files['filenames']
    list_of_items = list_of_items[list_of_items[SEGMENTS].isin(list_of_files) & list_of_items[CLASSES].isin(list_of_files)]
    list_of_items.reset_index(drop=True)

    return list_of_items

def save_list(save_path, list_of_items):
    list_of_items.to_csv(save_path, header=False, index=False, columns=[IMAGES, SEGMENTS, CLASSES])

def platgesbcn_list(folder_path, save_path=None, img_ext=None, seg_ext=None, cls_ext=None):
    list_of_items = _tabulate_files(folder_path, img_ext=img_ext, seg_ext=seg_ext, cls_ext=cls_ext)
    if save_path is not None : save_list(save_path, list_of_items)
    return list_of_items
       
        
"""
if __name__ == "__main__":
    # TODO: random seed fixing

    folder_path = '/mnt/c/Users/Ignasi/Downloads/platgesbcn2021/'
    #folder_path = '/mnt/c/Users/Ignasi/Downloads/platgesbcn2021_fine/images/useful/'
    #folder_path = '/mnt/c/Users/Ignasi/Downloads/platgesbcn2021_wide/images/useful/'
    VERBOSE = True
    img_ext = '.JPG'
    seg_ext = '.segments.pkl'
    cls_ext = '.classes.pkl'


    save_path_all = "./data_lists/platgesbcn2021_all.csv"
    save_path_train = "./data_lists/platgesbcn2021_train.csv"
    save_path_test = "./data_lists/platgesbcn2021_test.csv"
    probs = [0.7, 0.3]


    list_of_lists = platgesbcn_list(folder_path, save_path=save_path_all, img_ext=img_ext, seg_ext=seg_ext, cls_ext=cls_ext)

    train_list, test_list = split_pandas(list_of_lists, probs=probs)
    save_list(save_path_train, train_list)
    save_list(save_path_test, test_list)
"""
