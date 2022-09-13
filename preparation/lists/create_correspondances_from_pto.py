
# It creates a csv which can be used to generate the manual correspondances format from wnsayo_create_platja_models.ipynb
# It creates a json consistent with the ImagesGroupDataset

import re
import pandas as pd


FILENAME = 'filename'
IMG_1 = 'img_1'
IMG_2 = 'img_2'
X1 = 'x1'
Y1 = 'y1'
X2 = 'x2'
Y2 = 'y2'


RE_NOMBRE   = rf'n\"(?P<{FILENAME}>.*)\"'

RE_IMG_1    = rf'n(?P<{IMG_1}>\d*)'
RE_X1       = rf'x(?P<{X1}>\d*([.]\d*)?)'
RE_Y1       = rf'y(?P<{Y1}>\d*([.]\d*)?)'
RE_IMG_2    = rf'N(?P<{IMG_2}>\d*)'
RE_X2       = rf'X(?P<{X2}>\d*([.]\d*)?)'
RE_Y2       = rf'Y(?P<{Y2}>\d*([.]\d*)?)'


RE_IMAGE_LINE           = rf'^i(\s+({RE_NOMBRE}|\S*))*$'
RE_CORRESPONDANCE_LINE  = rf'^c(\s+({RE_IMG_1}|{RE_IMG_2}|{RE_X1}|{RE_Y1}|{RE_X2}|{RE_Y2}|\S*))*$'


def process_pto(pto_filepath, windows=True, windows_root=None):
    windows_root = windows_root or '/mnt/c/'

    regex = re.compile(rf'({RE_IMAGE_LINE}|{RE_CORRESPONDANCE_LINE})', flags=re.MULTILINE)

    # There should exist a better way to read the file, apply the regex and generate the table
    with open(pto_filepath, 'r') as f:
        data = [regex.search(x) for x in f]
    
    data = [m.groupdict() for m in data if m is not None]
    data_df = pd.DataFrame(data)

    # There is some better ways to do this:
    if windows:
        file_list = data_df.loc[~data_df[FILENAME].isnull(), FILENAME].str.replace(r'[A-Za-z]:\\', windows_root, regex=True).str.replace(r'\\', '/', regex=True).to_list()
    else:
        file_list = data_df.loc[~data_df[FILENAME].isnull(), FILENAME].to_list()

    correspondances_df = data_df.loc[data_df[FILENAME].isnull(), [IMG_1, X1, Y1, IMG_2, X2, Y2]].reset_index(drop=True).astype(float).astype(int)

    return correspondances_df, file_list
