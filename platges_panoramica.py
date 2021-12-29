
import cv2
import numpy as np
import sys
from torch.utils.data import DataLoader

from datasets.platges_homography_dataset import Platges_DronHomographyDataset
from docopts.help_panoramica import parse_args
from models.algorithms.homography import BasicStitching, RANSAC_MatrixFinder

from extern.SuperPointPretrainedNetwork.demo_superpoint import SuperPointFrontend


############ FOR DEBUGGING ############
NUM_IMGS = None # int or None

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

#######################################

POINT_FINDER_TYPES = ['SIFT', 'SUPERPOINTS', 'SIFT_HANDPICKED', 'HANDPICKED']


def buid_dataloader(cluster, time, longitude_bin_size, latitude_bin_size, min_imgs_bin, data_path, downsample=None):
    dataset = Platges_DronHomographyDataset(data_path, 
                                            to_tensor=False, 
                                            cluster=cluster, 
                                            th_time=time, 
                                            longitude_bin_size=longitude_bin_size, 
                                            latitude_bin_size=latitude_bin_size, 
                                            min_per_bin=min_imgs_bin,
                                            downsample=downsample)
    def my_collate(x): return x # <- do not transform imgs to tensor
    dataloader = DataLoader(dataset, collate_fn=my_collate)

    return dataloader

def build_pointfinder(point_finder_number, weights_path=None, nms_dist=None, conf_th=None, nn_th=None, radi=None, cuda=False):
    POINT_FINDER = POINT_FINDER_TYPES[point_finder_number]

    if POINT_FINDER == 'SUPERPOINTS':
        
        fe = SuperPointFrontend(weights_path=weights_path,
                                nms_dist=nms_dist,
                                conf_thresh=conf_th,
                                nn_thresh=nn_th,
                                cuda=cuda)

        def point_finder(img, other):
            # normalize image to "numpy float32 input image in range [0,1]".
            img = np.float32(img) / 255

            pts0, des = fe.run(img)[:2]
            pts = tuple(cv2.KeyPoint(int(round(p[0])), int(round(p[1])), int(round(p[2] * radi))) for p in pts0.T)

            return pts, des.T
        
        ARG2 = None

    elif POINT_FINDER == 'SIFT':

        descriptor = cv2.SIFT_create()
        point_finder = lambda img, other : descriptor.detectAndCompute(img, other)
        ARG2 = None

    elif POINT_FINDER == 'SIFT_HANDPICKED':

        descriptor = cv2.SIFT_create()
        point_finder = lambda img, kp : descriptor.compute(img, kp)#[::-1]
        ARG2 = HANDPICKED_KP
    
    elif POINT_FINDER == 'HANDPICKED':
        point_finder = None
        ARG2 = HANDPICKED_KP
    
    return point_finder, ARG2

def build_matrix_estimator(ransac_th):
    bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
    matrix_finder = RANSAC_MatrixFinder(point_matcher=bf,
                                        MIN_MATCH_COUNT=4,
                                        ransacReprojThreshold=ransac_th,
                                        maxIters=1000)
    return matrix_finder

def run_iter(group_of_imgs, metas, sticher, ARG2=None, verbose=False, plot=False):
    if verbose:
        print('\n'.join([f"{i} - {meta['path']}" for i, meta in enumerate(metas)]))
        print("\n---\n")

    if NUM_IMGS is not None:
        group_of_imgs = group_of_imgs[0:NUM_IMGS]

    panorama = sticher(group_of_imgs, verbose=verbose, plot=plot, descriptor_arg2=ARG2)

    return panorama


def main(cluster, time, longitude_bin_size, latitude_bin_size, min_imgs_bin, data_path, save_dir, downsample, point_finder_number, weights_path, nms_dist, conf_th, nn_th, radi, ransac_th, cuda, verbose, plot):

    # TODO: mask beach and/or water from buildings
    dataloader = buid_dataloader(cluster, time, longitude_bin_size, latitude_bin_size, min_imgs_bin, data_path, downsample)

    point_finder, ARG2 = build_pointfinder(point_finder_number, weights_path, nms_dist, conf_th, nn_th, radi, cuda)
    matrix_finder = build_matrix_estimator(ransac_th)

    sticher = BasicStitching(point_finder_descriptor=point_finder, homography_matrix_estimator=matrix_finder)

    for batch in dataloader:
        for group_of_imgs, metas in batch:
            panorama = run_iter(group_of_imgs, metas, sticher, ARG2, verbose, plot)
            cv2.imwrite(save_dir, panorama)


if __name__ == "__main__":
    # only on the "if __name__ == "__main__":" section it will be admited the *args notation if there is not a really great justification

    # Read args
    #cluster, time, longitude_bin_size, latitude_bin_size, min_imgs_bin, data_path, save_dir, downsample, point_finder_number, weights_path, nms_dist, conf_th, nn_th, radi, ransac_th, cuda, verbose, plot = parse_args(sys.argv[1:])
    args = parse_args(sys.argv[1:])

    # Run application
    #main(cluster, time, longitude_bin_size, latitude_bin_size, min_imgs_bin,
    #     data_path, save_dir, downsample, point_finder_number, weights_path, 
    #     nms_dist, conf_th, nn_th, radi, ransac_th, cuda, verbose, plot)
    main(*args)
    
    