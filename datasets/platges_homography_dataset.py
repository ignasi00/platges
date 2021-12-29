
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
import cv2
from datetime import datetime
from exif import Image
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset


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

class Platges_DronHomographyDataset(Dataset):

    def _list_metadata(self, folder_path, cluster=True):
        folder_path = os.path.abspath(folder_path)

        list_of_items = list()
        for f in os.scandir(folder_path):
            if f.is_file():
                try:
                    with open(f.path, 'rb') as src:

                        img = Image(src)
                        if img.has_exif:
                            timestamp = (datetime.strptime(img.datetime_original, '%Y:%m:%d %H:%M:%S') - TIME_ZERO).total_seconds()
                            gps_to_sec = lambda gps : 3600 * gps[0] + 60 * gps[1] + gps[0]
                            
                            list_of_items.append({
                                PATH        : f.path,
                                LONGITUDE   : gps_to_sec(img.gps_longitude),
                                LATITUDE    : gps_to_sec(img.gps_latitude),
                                TIMESTAMP   : timestamp,
                                FOCAL       : img.focal_length
                            })
                            if cluster == False : list_of_items[-1][HOMOGROUP] = 0

                except:
                    if cluster == False : list_of_items.append({PATH : f.path, HOMOGROUP : 0})
                    continue
        return list_of_items

    def _correct_clusters(self, pandas_table, serie, th_time=60, longitude_bin_size=180, latitude_bin_size=180, min_per_bin=1):
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

    def _cluster_images(self, pandas_table, th_time=60, longitude_bin_size=0.05, latitude_bin_size=0.05, min_per_bin=1):
        # 3 1D histogram bin reference
        pandas_table[TIMEGROUP] = (pandas_table[TIMESTAMP] - pandas_table[TIMESTAMP].min()) // th_time
        pandas_table[LONGGROUP] = (pandas_table[LONGITUDE] - pandas_table[LONGITUDE].min()) // longitude_bin_size
        pandas_table[LATIGROUP] = (pandas_table[LATITUDE] - pandas_table[LATITUDE].min()) // latitude_bin_size

        # 3D histogram
        serie = pandas_table.groupby([TIMEGROUP, LONGGROUP, LATIGROUP]).size()

        # for each lonely case, search if it could fit in a near bin
        pandas_table = self._correct_clusters(pandas_table, serie, th_time=th_time, longitude_bin_size=longitude_bin_size, latitude_bin_size=latitude_bin_size, min_per_bin=min_per_bin)

        # Group by bins (mapping combinations to indexes)
        group_map = pd.DataFrame(serie.keys().to_list(), columns=serie.keys().names)

        def map_rule(row, map_table=group_map):
            compare = lambda row, table, key : table[key] == row[key]
            return map_table[ (compare(row, map_table, TIMEGROUP)) \
                            & (compare(row, map_table, LONGGROUP)) \
                            & (compare(row, map_table, LATIGROUP))].index[0]

        pandas_table[HOMOGROUP] = pandas_table.apply(map_rule, axis=1)

        return pandas_table

    def __init__(self, folder_path, to_tensor=True, downsample=None, cluster=True, th_time=60, longitude_bin_size=0.05, latitude_bin_size=0.05, min_per_bin=1, read_flag=0):
        list_of_items = self._list_metadata(folder_path, cluster=cluster)
        self.table_of_items = pd.DataFrame(list_of_items)
        if cluster : self.table_of_items = self._cluster_images(self.table_of_items, th_time=th_time, min_per_bin=min_per_bin)
        
        self.read_flag = read_flag

        self.to_tensor = A.Compose([ToTensor()]) if to_tensor else None
        self.downsample = downsample

    def __getitem__(self, idx):
        # https://www.kaggle.com/yukia18/opencv-vs-pil-speed-comparisons-for-pytorch-user
        meta = list(self.table_of_items[self.table_of_items[HOMOGROUP] == idx].T.to_dict().values())
        images = [None] * len(meta)
        for i, dict_ in enumerate(meta):
            # TODO: read in color
            images[i] = cv2.imread(dict_[PATH], self.read_flag)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # <- currently is in grayscale
            # downsample TODO: improve style
            if self.downsample is not None and self.downsample != 1:
                images[i] = cv2.resize(images[i], (images[i].shape[1] // self.downsample, images[i].shape[0] // self.downsample), interpolation=cv2.INTER_AREA)
            # TODO: Color BGR to RGB if needed/works
            if self.to_tensor is not None:
                images[i] = self.to_tensor(image=images[i])['image']

        return images, meta

    def __len__(self):
        return self.table_of_items[HOMOGROUP].nunique()
