
import contextlib
import h5py
import numpy as np


IMAGES = 'images'
MASKS = 'masks'
HOMOGRAPHIES = 'homographies'
GPS = 'gps'
NAME = 'name'

FILENAME = 'filename'


def load_map_model(filename):
    with h5py.File(filename, 'r') as f:
        images = f[IMAGES][:]
        masks = f[MASKS][:]
        homographies = f[HOMOGRAPHIES][:]
        gps = f[GPS][:]
        name = f.attrs[NAME].value

        map_model = MapModel(images, masks, homographies, gps, name)
    
    return map_model

@contextlib.contextmanager
def context_map_model(filename):
    try:
        f = h5py.File(filename, 'r')

        images = f[IMAGES]
        masks = f[MASKS]
        homographies = f[HOMOGRAPHIES]
        gps = f[GPS]
        name = f.attrs[NAME]

        map_model = MapModel(images, masks, homographies, gps, name)
        yield map_model
    finally:
        f.close()

class MapModel():

    def __init__(self, images, masks, homographies, gps, name):
        
        self.images = images
        self.masks = masks
        self.homographies = homographies
        self.gps = gps
        self.name = name

    def __iter__(self):
        for img, mask, mat, gps in zip(self.images, self.masks, self.homographies, self.gps):
            yield img, mask, mat, gps
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.homographies[idx], self.gps[idx]

    def get_name(self):
        return self.name

    def get_mapping(self):
        return tuple(self) # return tuple(zip(self.images, self.masks, self.homographies, self.gps))
    
    def get_elements(self):
        return self.images, self.masks, self.homographies, self.gps
    
    def save_map_model(self, name=None, root='./'):

        images, masks, homographies, gps = self.get_elements()
        name = name or map_model.get_name()

        filename = f'{root}/{name}.hdf5'

        with h5py.File(filename, 'w') as f:
            f.create_dataset(IMAGES, images)
            f.create_dataset(MASKS, masks)
            f.create_dataset(HOMOGRAPHIES, homographies)
            f.create_dataset(GPS, gps)
            f.attrs[NAME] = name

def gps_search_map_model(gps, map_model_filename_list, limit=0):
    gps = np.asarray(gps) # shape 2 x 1
    assert gps.shape == (2, )

    best = None
    best_score = np.inf
    for filename in map_model_filename_list:
        with context_map_model(filename) as map_model:
            model_gps = np.array(map_model[GPS]) # shape N x 2
            score = np.min(np.norm(model_gps - gps, axis=1))
            
            if score < best_score:
                best_score = score
                best = filename
        
        if best_score <= limit : break
    return best
