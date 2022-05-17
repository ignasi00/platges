
import cv2
import numpy as np
import pathlib
import sys
from types import SimpleNamespace
from torch.utils.data import DataLoader

from frameworks.pytorch.datasets.images_group_dataset import ImagesGroupDataset
from docopts.help_panoramica import parse_args
from frameworks.opencv.homography.homography_matrix import RANSAC_MatrixFinder, BatchPairedHomographyMatrixEstimator, UnmatchedKeypoints_PointFinder, UnmatchedDescriptors_Descriptor
from frameworks.opencv.homography.stiching import compute_stiching

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

    if point_finder_name == "SUPERPOINTS":
        model = SuperPointFrontend(weights_path=weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=False)
        point_finder = lambda img, mask : mask
        def descriptor_finder(img, kp):
            pts, des = apply_superpoints_eval(model, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), radi)
        
        if kp is not None:
            in_mask = np.asarray([p.pt in np.argwhere(kp != 0) for p in pts])
            return pts[in_mask], des[in_mask]
        return pts, des

        bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
        matrix_finder = RANSAC_MatrixFinder(point_matcher=bf, min_match_count=4, ransacReprojThreshold=ransac_th, maxIters=ransac_maxIters)

    elif point_finder_name == 'SIFT':
        finder_obj = cv2.SIFT_create()
        point_finder = lambda img, mask : finder_obj.detect(img, mask)
        descriptor_finder = lambda img, kp : finder_obj.compute(img, kp)
        
        bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
        matrix_finder = RANSAC_MatrixFinder(point_matcher=bf, min_match_count=4, ransacReprojThreshold=ransac_th, maxIters=ransac_maxIters)

    elif point_finder_name == 'SIFT_HANDPICKED':
        id_keypoints_dict = {i : kps for i, kps in enumerate(handpicked)}
        id_list = list(range(len(handpicked)))
        point_finder = UnmatchedKeypoints_PointFinder(lambda x : id_list.pop(0), id_keypoints_dict)

        finder_obj = cv2.SIFT_create()
        descriptor_finder = lambda img, kp : finder_obj.compute(img, kp)

        bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
        matrix_finder = RANSAC_MatrixFinder(point_matcher=bf, min_match_count=4, ransacReprojThreshold=ransac_th, maxIters=ransac_maxIters)
    
    elif point_finder_name == 'HANDPICKED':
        id_keypoints_dict = {i : kps for i, kps in enumerate(handpicked)}
        id_list = list(range(len(handpicked)))
        point_finder = UnmatchedKeypoints_PointFinder(lambda x : id_list.pop(0), id_keypoints_dict)
        
        id_descriptors_dict = {i : v_des[i] for i, img in enumerate(handpicked)}
        id_list2 = list(range(len(handpicked)))
        descriptor_finder = UnmatchedDescriptors_Descriptor(lambda x : id_list2.pop(0), id_descriptors_dict)

        matrix_finder = MatchedKeypoints_RANSAC_MatrixEstimator(min_match_count=4, ransacReprojThreshold=ransac_th, maxIters=ransac_maxIters)
    
    return point_finder, descriptor_finder, matrix_finder


def main(list_path, point_finder_name, data_root='', outputs_root='', downsample=1, handpicked=None, weights_path=None, nms_dist=None, conf_th=None, nn_th=None, radi=None, ransac_th=5.0, ransac_maxIters=1000, num_imgs=None, VERBOSE=False, VERBOSE_TQDM=False, VERBOSE_MAT=False, VERBOSE_PLOT=False):

    dataset = ImagesGroupDataset(json_path=list_path, data_root=data_root, downsample=downsample)
    def my_collate(x): # <- x is a batch like [dataset[i] for i in batched_indices]
        return x # <- do not transform imgs to tensor
    dataloader = DataLoader(dataset, collate_fn=my_collate)


    point_finder, descriptor_finder, matrix_finder = build_model_parameters( point_finder_name, 
                                                                            handpicked=handpicked,
                                                                            weights_path=weights_path,
                                                                            nms_dist=nms_dist,
                                                                            conf_th=conf_th,
                                                                            nn_th=nn_th,
                                                                            radi=radi,
                                                                            ransac_th=ransac_th,
                                                                            ransac_maxIters=ransac_maxIters,
                                                                            cuda=False)

    model = BatchPairedHomographyMatrixEstimator(point_finder=point_finder, descriptor_finder=descriptor_finder, matrix_finder=matrix_finder)

    n_saved = 0
    for batch in dataloader: # each batch is a [group] * N
        for group in batch: # each group is a tuple with (imgs, path)

            group_of_imgs, pathes = tuple(zip(*list(group)))

            if VERBOSE:
                print("\n---\n")
                print('\n'.join([f"{i} - {path}" for i, path in enumerate(pathes)]))
                print("\n---\n")
            
            if num_imgs is not None:
                group_of_imgs = group_of_imgs[0:num_imgs]

            H, _, _, n_matches = model(group_of_imgs, base_idx=0)
            panorama = compute_stiching(group_of_imgs, H, n_matches, base_idx=0, matrix_applier=None, blender=None)
            
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