
import cv2
import numpy as np
import pathlib
import sys
from types import SimpleNamespace
from torch.utils.data import DataLoader

from framework.pytorch.datasets.images_group_dataset import ImagesGroupDataset
from docopts.help_panoramica import parse_args
from framework.opencv.overlap_finder import BatchOverlapFinder, RANSAC_MatrixFinder

from extern.superpoints_utils import SuperPointFrontend
from extern.superpoints_utils import apply_net_eval as apply_superpoints_eval


experiment_params = SimpleNamespace(
    list_path = './data_lists/platges2021_all.json',
    data_root = '', #'/mnt/c/Users/Ignasi/Downloads/PlatgesBCN/dron/'
    outputs_root = './outputs/panoramica_platges_GPS/',
    downsample = 4,
    point_finder_name = 'SIFT',
    # HANDPICKED: it is implemented but not available because it requieres some hard-wireing (better config files).
    # HANDPICKED: The best solution would be implement a handpicked model and implement the hand picked as a dataset (with its data_list).
    handpicked = None,
    weights_path = 'extern/SuperPointPretrainedNetwork/superpoint_v1.pth',
    nms_dist = 4,
    conf_th = 0.015,
    nn_th = 0.7,
    radi = 2,
    ransac_th = 5.0,
    ransac_maxIters = 1000,
    num_imgs = None
)


def build_model_parameters(point_finder_name, *, handpicked=None, weights_path=None, nms_dist=None, conf_th=None, nn_th=None, radi=None, ransac_th=5.0, ransac_maxIters=1000, cuda=False):
    # TODO: separate point_finder from homography_estimator

    if point_finder_name == "SUPERPOINTS":
        model = SuperPointFrontend(weights_path=weights_path, nms_dist=nms_dist, conf_thresh=conf_th, nn_thresh=nn_th, cuda=cuda)
        point_finder_descriptor = lambda img, other : apply_superpoints_eval(model, img, radi)
        descriptor_arg2 = None

        bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
        matrix_finder = RANSAC_MatrixFinder(point_matcher=bf, MIN_MATCH_COUNT=4, ransacReprojThreshold=ransac_th, maxIters=ransac_maxIters)
        homography_matrix_estimator = matrix_finder

    elif point_finder_name == 'SIFT':
        descriptor = cv2.SIFT_create()
        point_finder_descriptor = lambda img, other : descriptor.detectAndCompute(img, other)
        descriptor_arg2 = None
        
        bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
        matrix_finder = RANSAC_MatrixFinder(point_matcher=bf, MIN_MATCH_COUNT=4, ransacReprojThreshold=ransac_th, maxIters=ransac_maxIters)
        homography_matrix_estimator = matrix_finder

    elif point_finder_name == 'SIFT_HANDPICKED':
        descriptor = cv2.SIFT_create()
        point_finder_descriptor = lambda img, kp : descriptor.compute(img, kp)#[::-1]
        descriptor_arg2 = handpicked or tuple()

        bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
        matrix_finder = RANSAC_MatrixFinder(point_matcher=bf, MIN_MATCH_COUNT=4, ransacReprojThreshold=ransac_th, maxIters=ransac_maxIters)
        homography_matrix_estimator = matrix_finder 
    
    elif point_finder_name == 'HANDPICKED':
        point_finder_descriptor = lambda img, other : (handpicked or tuple(), None)
        descriptor_arg2 = None
        def homography_matrix_estimator(kp1, des1, kp2, des2):
            kp1 = np.float32([kp.pt for kp in kp1]).reshape(-1, 1, 2)
            kp2 = np.float32([kp.pt for kp in kp2]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC, ransac_th, maxIters=ransac_maxIters)
            return H, None, mask
    
    return point_finder_descriptor, descriptor_arg2, homography_matrix_estimator


def main(list_path, point_finder_name, data_root='', outputs_root='', downsample=1, handpicked=None, weights_path=None, nms_dist=None, conf_th=None, nn_th=None, radi=None, ransac_th=5.0, ransac_maxIters=1000, num_imgs=None, VERBOSE=False, VERBOSE_TQDM=False, VERBOSE_MAT=False, VERBOSE_PLOT=False):

    dataset = ImagesGroupDataset(json_path=list_path, data_root=data_root, downsample=downsample)
    def my_collate(x): # <- x is a batch like [dataset[i] for i in batched_indices]
        return x # <- do not transform imgs to tensor
    dataloader = DataLoader(dataset, collate_fn=my_collate)


    point_finder_descriptor, descriptor_arg2, homography_matrix_estimator = build_model_parameters( point_finder_name, 
                                                                                                    handpicked=handpicked,
                                                                                                    weights_path=weights_path,
                                                                                                    nms_dist=nms_dist,
                                                                                                    conf_th=conf_th,
                                                                                                    nn_th=nn_th,
                                                                                                    radi=radi,
                                                                                                    ransac_th=ransac_th,
                                                                                                    ransac_maxIters=ransac_maxIters,
                                                                                                    cuda=False)

    model = BatchOverlapFinder(point_finder_descriptor=point_finder_descriptor, homography_matrix_estimator=homography_matrix_estimator, training=True)


    n_saved = 0
    for batch in dataloader:
        for group in batch: # batch is list of (img, path) elements

            group_of_imgs, pathes = tuple(zip(*list(group)))

            if VERBOSE:
                print("\n---\n")
                print('\n'.join([f"{i} - {path}" for i, path in enumerate(pathes)]))
                print("\n---\n")
            
            if num_imgs is not None:
                group_of_imgs = group_of_imgs[0:num_imgs]

            panorama, _ = model(group_of_imgs, descriptor_arg2=descriptor_arg2, base_idx=0, VERBOSE_TQDM=VERBOSE_TQDM, VERBOSE_MAT=VERBOSE_MAT, VERBOSE_PLOT=VERBOSE_PLOT)
            cv2.imwrite(f"{outputs_root}/{n_saved}.jpg", panorama)
            n_saved += 1


if __name__ == "__main__":

    (experiment_params, verbose_main, verbose_tqdm, verbose_mat, verbose_plot) = parse_args(sys.argv)

    ###############################################################################

    pathlib.Path(experiment_params.outputs_root).mkdir(parents=True, exist_ok=True)

    main(   experiment_params.list_path,
            experiment_params.point_finder_name,
            experiment_params.data_root,
            experiment_params.outputs_root,
            experiment_params.downsample,
            experiment_params.handpicked,
            experiment_params.weights_path,
            experiment_params.nms_dist,
            experiment_params.conf_th,
            experiment_params.nn_th,
            experiment_params.radi,
            experiment_params.ransac_th,
            experiment_params.ransac_maxIters,
            experiment_params.num_imgs,
            VERBOSE=verbose_main, VERBOSE_TQDM=verbose_tqdm, VERBOSE_MAT=verbose_mat, VERBOSE_PLOT=verbose_mat)