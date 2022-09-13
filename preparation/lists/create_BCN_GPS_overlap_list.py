
from datetime import datetime
from exif import Image
import json
import numpy as np
import os
import pandas as pd

from .utils.split_list import split_list


TIME_ZERO = datetime(1970,1,1)

PATH = 'path'
LONGITUDE = 'longitude'
LATITUDE = 'latitude'
TIMESTAMP = 'timestamp'
FOCAL = 'focal'

TIMEGROUP = 'timegroup'
LONGGROUP = 'longitude_group'
LATIGROUP = 'latitude_group'
HOMOGROUP = 'group'

PATHS = 'paths'


def _list_metadata(folder_path, verbose=False):
    folder_path = os.path.abspath(folder_path)

    list_of_items = list()
    for f in os.scandir(folder_path):
        if f.is_file():
            try:
                with open(f.path, 'rb') as src:

                    img = Image(src)
                    if img.has_exif:
                        timestamp = (datetime.strptime(img.datetime_original, '%Y:%m:%d %H:%M:%S') - TIME_ZERO).total_seconds()
                        gps_to_sec = lambda gps : gps[0] + gps[1] / 60 + gps[2] / 3600
                        
                        list_of_items.append({
                            PATH        : f.path,
                            LONGITUDE   : gps_to_sec(img.gps_longitude),
                            LATITUDE    : gps_to_sec(img.gps_latitude),
                            TIMESTAMP   : timestamp,
                            FOCAL       : img.focal_length
                        })
            except:
                if verbose: print(f"SKIP: {f.path}")
                continue
    return list_of_items

def _correct_clusters(pandas_table, serie, th_time=60, longitude_bin_size=180, latitude_bin_size=180, min_per_bin=1):
    # Lonely bins
    lonely = serie[serie.ge(1).le(min_per_bin)].keys().to_list()

    # for each lonely case, search if it could fit in a near bin
    for time, longi, lati in lonely:
        try:
            in_range = lambda table, key, value, range_ : (np.abs(table[key] - value) < range_ / 2)
            src_idx = pandas_table[ (in_range(pandas_table, TIMEGROUP, time, th_time)) \
                                    & (in_range(pandas_table, LONGGROUP, longi, longitude_bin_size)) \
                                    & (in_range(pandas_table, LATIGROUP, lati, latitude_bin_size)) \
                                    & ((pandas_table[LONGGROUP] != longi) | (pandas_table[LATIGROUP] != lati))].index[0]

            dst_idx = pandas_table[(pandas_table[TIMEGROUP] == time) & (pandas_table[LONGGROUP] == longi) & (pandas_table[LATIGROUP] == lati)].index[0]
            pandas_table.loc[dst_idx, TIMEGROUP] = pandas_table.loc[src_idx, TIMEGROUP]
            pandas_table.loc[dst_idx, LONGGROUP] = pandas_table.loc[src_idx, LONGGROUP]
            pandas_table.loc[dst_idx, LATIGROUP] = pandas_table.loc[src_idx, LATIGROUP]
        except:
            continue
    
    return pandas_table

def _cluster_images(pandas_table, th_time=60, longitude_bin_size=0.05, latitude_bin_size=0.05, min_per_bin=1):
    # 3 1D histogram bin reference
    pandas_table[TIMEGROUP] = (pandas_table[TIMESTAMP] - pandas_table[TIMESTAMP].min()) // th_time
    pandas_table[LONGGROUP] = (pandas_table[LONGITUDE] - pandas_table[LONGITUDE].min()) // longitude_bin_size
    pandas_table[LATIGROUP] = (pandas_table[LATITUDE] - pandas_table[LATITUDE].min()) // latitude_bin_size

    # 3D histogram
    serie = pandas_table.groupby([TIMEGROUP, LONGGROUP, LATIGROUP]).size()

    # for each lonely case, search if it could fit in a near bin
    pandas_table = _correct_clusters(pandas_table, serie, th_time=th_time, longitude_bin_size=longitude_bin_size, latitude_bin_size=latitude_bin_size, min_per_bin=min_per_bin)

    # Group by bins (mapping combinations to indexes)
    group_map = pd.DataFrame(serie.keys().to_list(), columns=serie.keys().names)

    def map_rule(row, map_table=group_map):
        compare = lambda row, table, key : table[key] == row[key]
        return map_table[ (compare(row, map_table, TIMEGROUP)) \
                        & (compare(row, map_table, LONGGROUP)) \
                        & (compare(row, map_table, LATIGROUP))].index[0]

    pandas_table[HOMOGROUP] = pandas_table.apply(map_rule, axis=1)

    return pandas_table

def save_list(save_path, list_of_lists):
    with open(save_path, 'w') as f: 
        json.dump(list_of_lists, f)

def EXIF_homography_list(folder_path, save_path=None, th_time=60, longitude_bin_size=0.05, latitude_bin_size=0.05, min_per_bin=1, verbose=False):
    list_of_items = _list_metadata(folder_path, verbose=verbose)
    table_of_items = pd.DataFrame(list_of_items)
    table_of_items = _cluster_images(table_of_items, th_time=th_time, longitude_bin_size=longitude_bin_size, latitude_bin_size=latitude_bin_size, min_per_bin=min_per_bin)

    list_of_lists = [table_of_items[PATH].values[table_of_items[HOMOGROUP] == idx].tolist() for idx in table_of_items[HOMOGROUP].unique()]
    #list_of_lists = [{PATHS : list_} for list_ in list_of_lists]
    
    if save_path is not None : save_list(save_path, list_of_lists)
        
    return list_of_lists


"""
if __name__ == "__main__":
    # TODO: random seed fixing

    folder_path = '/mnt/c/Users/Ignasi/Downloads/PlatgesBCN/dron'
    VERBOSE = True
    th_time = 60
    longitude_bin_size = 0.05
    latitude_bin_size = 0.05
    min_per_bin = 1


    save_path_all = "./data_lists/platges2021_all.json"
    save_path_train = "./data_lists/platges2021_train.json"
    save_path_test = "./data_lists/platges2021_test.json"
    probs = [0.7, 0.3]


    list_of_lists = EXIF_homography_list(folder_path, save_path=save_path_all, 
                                        th_time=th_time, longitude_bin_size=longitude_bin_size, 
                                        latitude_bin_size=latitude_bin_size, min_per_bin=min_per_bin, verbose=VERBOSE)

    train_list, test_list = split_list(list_of_lists, probs=probs)
    save_list(save_path_train, train_list)
    save_list(save_path_test, test_list)
"""
