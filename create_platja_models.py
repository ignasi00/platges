
# From a .json (list of images path where order is relevant) and a .csv that relate two indexes of the json list with two pairs of coordiantes, build a map model with all it requiers.
# The .csv columns are img_idx_1, x1, y1, img_idx_2, x2, y2
# The masks should have the same name as the image, but they have different extension. masks_path points to a folder

import cv2
from exif import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import sys

from docopts.help_create_platja_models import parse_args
from frameworks.opencv.homography.homography_matrix import UnmatchedKeypoints_PointFinder, UnmatchedDescriptors_Descriptor, RANSAC_MatrixEstimator, BatchHomographyMatrixEstimator
from frameworks.opencv.homography.stiching import make_stiching, color_matrix_applier, color_blender
from frameworks.pytorch.datasets.images_group_dataset import ImagesGroupDataset
from platges_utils.datasets.platgesbcn_segmentation_dataset import PlatgesBCNSegmentationDataset
from platges_utils.map_model import MapModel, context_map_model
from platges_utils.map_model_utils import gps_search_map_model



IMG1 = 'img1'
X1 = 'x1'
Y1 = 'y1'
IMG2 = 'img2'
X2 = 'x2'
Y2 = 'y2'


def load_data(json_path, csv_path, data_root=''):
    dataset = ImagesGroupDataset(json_path=json_path, data_root=data_root, downsample=None, read_flag=cv2.IMREAD_COLOR)
    correspondances_df = pd.read_csv(csv_path, header=None, index_col=False, names=[IMG1, X1, Y1, IMG2, X2, Y2])
    return dataset, correspondances_df

def generate_correspondances(correspondances_df):
    # Hugin pto left column is left image at the editor and right column is right image at the editor.
    # This function expect that the images at the editor follows the real order, additionally, the list of images at Hugins must follow the correct sequence.
    # It also assumes that there is at most 2 neighbours, one at each side
    
    handpicked_kps = list()
    handpicked_descriptors = list()
    
    images = np.unique(correspondances_df[[IMG1, IMG2]].to_numpy().flatten())
    for img in images:
        left_points = correspondances_df[correspondances_df[IMG1] == img][[X1, Y1]].values.tolist()
        right_points = correspondances_df[correspondances_df[IMG2] == img][[X2, Y2]].values.tolist()

        handpicked_kps.append(cv2.KeyPoint_convert(left_points + right_points))
        handpicked_descriptors.append([(img, 1)] * len(left_points) + [(img, 0)] * len(right_points))
    
    return handpicked_kps, handpicked_descriptors


def create_homographies(correspondances_df, v_img):

    handpicked_kps, handpicked_descriptors = generate_correspondances(correspondances_df)
    id_keypoints_dict = {i : kps for i, kps in enumerate(handpicked_kps)}
    id_descriptors_dict = {i : des for i, des in enumerate(handpicked_descriptors)}

    id_list = list(range(len(handpicked_kps)))
    point_finder = UnmatchedKeypoints_PointFinder(lambda x : id_list.pop(0), id_keypoints_dict)

    id_list2 = list(range(len(handpicked_descriptors)))
    descriptor_finder = UnmatchedDescriptors_Descriptor(lambda x : id_list2.pop(0), id_descriptors_dict)

    # _matrix_finder is a function defined below
    homographies_estimator = BatchHomographyMatrixEstimator(point_finder=point_finder, descriptor_finder=descriptor_finder, matrix_finder=_matrix_finder)
    v_homographies, _, _, _ = homographies_estimator(v_img)

    return v_homographies

def create_model(model_name, json_path, csv_path, masks_path, output_path=None, data_root=''):
    # The masks should have the same name as the image, but they have different extension. masks_path points to a folder
    dataset, correspondances_df = load_data(json_path, csv_path, data_root=data_root)

    v_img = []
    v_masks = []
    #v_homographies = []
    v_gps = []

    for d in list(dataset[0]):
    
        v_img.append(d[0])

        with open(d[1], 'rb') as src:
            img = Image(src)
            gps_to_sec = lambda gps : gps[0] + gps[1] / 60 + gps[2] / 3600
            v_gps.append((gps_to_sec(img.gps_latitude), gps_to_sec(img.gps_longitude)))
            
        with open(f'{masks_path}/{os.path.basename(d[1])[:-4]}.segments.pkl', 'rb') as f:
            mask = pickle.load(f, encoding='latin1')
            v_masks.append(mask)
    
    v_homographies = create_homographies(correspondances_df, v_img)

    map_model = MapModel(v_img, v_masks, v_homographies, v_gps, model_name)
    if output_path is not None : map_model.save_map_model(root=output_path)
    
    return map_model


def _point_matcher(des1, des2, distance=10):
    # desX have vectors of tuples: current_image_idx (smaller on the right size, higher on the lft size) and 0/1 (to the left correspondance == 0 and to the right correspondance == 1)
    left, right, order = (des1, des2, 1) if des1[0][0] > des2[0][0] else (des2, des1, 0)
    
    idx_left_to_right = np.argwhere(np.asarray(left)[:, 1] == 1).flatten().tolist()
    idx_right_to_left = np.argwhere(np.asarray(right)[:, 1] == 0).flatten().tolist()
    
    # return [cv2.DMatch(i, j, distance) if order else cv2.DMatch(j, i, distance) for i, j in zip(idx_left_to_right, idx_right_to_left)]
    if order:
        return [cv2.DMatch(i, j, distance) for i, j in zip(idx_left_to_right, idx_right_to_left)]
    else:
        return [cv2.DMatch(i, j, distance) for i, j in zip(idx_right_to_left, idx_left_to_right)]

# LMS instead of RANSAC
def _matrix_finder(kp1, des1, kp2, des2):
    matches = _point_matcher(des1, des2)
    
    src_pts12 = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
    dst_pts12 = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts12, dst_pts12)
    return H, matches, mask

# _matrix_finder = RANSAC_MatrixEstimator(point_matcher=_point_matcher, min_match_count=4, ransacReprojThreshold=116.1, maxIters=None)


if __name__ == "__main__":

    (model_name, json_path, csv_path, masks_path, output_path) = parse_args(sys.argv)

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    create_model(model_name, json_path, csv_path, masks_path, output_path=output_path)
