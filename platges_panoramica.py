
import cv2
import numpy as np
from torch.utils.data import DataLoader

from datasets.platges_homography_dataset import Platges_DronHomographyDataset
from models.algorithms.homography import BasicStitching, RANSAC_MatrixFinder

from extern.SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend


POINT_FINDER_TYPES = ['SIFT', 'SUPERPOINTS', 'HANDPICKED']
############# TODO: DOCOPT #############
DATA_PATH = '/mnt/c/Users/Ignasi/Downloads/4fotos' #'/mnt/c/Users/Ignasi/Downloads/PlatgesBCN/dron'
SAVE_PATH = '/mnt/c/Users/Ignasi/Downloads/panorama_test.jpg'
DOWNSAMPLE = 4
CLUSTER = True
NUM_IMGS = 2 # int or None

POINT_FINDER = POINT_FINDER_TYPES[1]
RANSAC_TH = 5

VERBOSE = True
PLOT = False
############# ##### ###### #############

RADI = 2
HANDPICKED_KP = (
    (
        cv2.KeyPoint(129, 1470, RADI),
        cv2.KeyPoint(262, 1458, RADI),
        cv2.KeyPoint(118, 1504, RADI),
        cv2.KeyPoint(288, 1491, RADI),
        cv2.KeyPoint(301, 1452, RADI),
        cv2.KeyPoint(94, 1471, RADI),
        cv2.KeyPoint(845, 1335, RADI),
        cv2.KeyPoint(705, 1345, RADI),
        cv2.KeyPoint(1044, 1239, RADI)
    ),
    (
        cv2.KeyPoint(5179, 1269, RADI),
        cv2.KeyPoint(5259, 1264, RADI),
        cv2.KeyPoint(5381, 1286, RADI),
        cv2.KeyPoint(5313, 1260, RADI),
        cv2.KeyPoint(5267, 1291, RADI),
        cv2.KeyPoint(5161, 1270, RADI),
        cv2.KeyPoint(5383, 1198, RADI),
        cv2.KeyPoint(5298, 1203, RADI),
        cv2.KeyPoint(5263, 1134, RADI)
    )
)


if __name__ == "__main__":
    
    dataset = Platges_DronHomographyDataset(DATA_PATH, 
                                            to_tensor=False, 
                                            cluster=CLUSTER, 
                                            th_time=60, 
                                            longitude_bin_size=180, 
                                            latitude_bin_size=180, 
                                            min_per_bin=1,
                                            downsample=DOWNSAMPLE)
    def my_collate(x): return x # <- do not transform imgs to tensor
    dataloader = DataLoader(dataset, collate_fn=my_collate)

    # TODO: mask beach and/or water from cityscapes


    if POINT_FINDER == 'SUPERPOINTS':
        
        fe = SuperPointFrontend(weights_path='extern/SuperPointPretrainedNetwork/superpoint_v1.pth',
                            nms_dist=4,
                            conf_thresh=0.015,
                            nn_thresh=0.7,
                            cuda=False)

        # normalize image to "numpy float32 input image in range [0,1]".
        def point_finder(img, other):
            img = np.float32(img) / 255

            pts0, des = fe.run(img)[:2]
            pts = tuple(cv2.KeyPoint(int(round(p[0])), int(round(p[1])), int(round(p[2] * RADI))) for p in pts0.T)

            return pts, des.T
        
        ARG2 = None

    elif POINT_FINDER == 'SIFT':

        descriptor = cv2.SIFT_create()
        point_finder = lambda img, kp : descriptor.compute(img, kp)#[::-1]
        ARG2 = None
    
    elif POINT_FINDER == 'HANDPICKED':
        point_finder = None
        ARG2 = HANDPICKED_KP


    bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
    matrix_finder = RANSAC_MatrixFinder(point_matcher=bf,
                                        MIN_MATCH_COUNT=4,
                                        ransacReprojThreshold=RANSAC_TH,
                                        maxIters=1000)
    sticher = BasicStitching(point_finder_descriptor=point_finder, homography_matrix_estimator=matrix_finder)

    for batch in dataloader:
        for imgs, metas in batch:
            if VERBOSE:
                print('\n'.join([f"{i} - {meta['path']}" for i, meta in enumerate(metas)]))
                print("\n---\n")

            if NUM_IMGS is not None:
                imgs = imgs[0:NUM_IMGS]

            panorama = sticher(imgs, verbose=VERBOSE, plot=PLOT, descriptor_arg2=ARG2)

            cv2.imwrite(SAVE_PATH, panorama)

            raise
