
# Given images and homography matrix, generate the stiched final image

import cv2
import numpy as np

from .utils import compute_dimensions, compute_path_homography, _image_idx_iterator


def default_matrix_applier(img, h, size):
    return cv2.warpPerspective(img, h, size, flags=cv2.INTER_LINEAR)

def color_matrix_applier(img, h, size):
    try:
        C = img.shape[2]
    except:
        return default_matrix_applier(img, h, size)

    img_C = np.zeros((size[1], size[0], C))
    for i in range(C):
        img_C[:, :, i] = cv2.warpPerspective(img[:, :, i].copy(), h, size, flags=cv2.INTER_LINEAR)
    
    return img_C

def default_blender(composed_img, current_img):
    composed_img[composed_img == 0] = current_img[composed_img == 0]
    return composed_img

def color_blender(composed_img, current_img):
    try:
        C = composed_img.shape[2]
    except:
        assert len(current_img.shape) == len(composed_img.shape)
        return default_blender(composed_img, current_img)

    img_C = composed_img.copy()
    for i in range(C):
        img_C[composed_img[:, :, ..., i] == 0, ..., i] = current_img[composed_img[:, :, ..., i] == 0, ..., i]

    return img_C

def compute_stiching(x, H, n_matches, base_idx=0, matrix_applier=None, blender=None):
    # Stitch as much images over the base_idx image plane as posible

    matrix_applier = matrix_applier or default_matrix_applier
    blender = blender or default_blender

    composed_img = x[base_idx].copy()
    # Translation of the base plane from previous stiched images
    translation = np.eye(3, 3)

    for current_idxs in _image_idx_iterator(n_matches, base_idx):
        current_img = x[current_idxs[-1]].copy()

        # 1- compute the homography matrix for the current image (chained homographies with previous translations)
        homography_matrix = compute_path_homography(current_idxs, H)
        if homography_matrix is None : break # _image_idx_iterator implements a "most likely matched first" idea
        homography_matrix = translation * homography_matrix

        # 2- compute the new dimensions
        max_x, min_x, max_y, min_y = compute_dimensions(composed_img, current_img, homography_matrix)
        width = int(max_x - min_x + 1)
        height = int(max_y - min_y + 1)

        # 3- compute the transformation matrix for the current image (homography matrix from step 1 with current trtanslation)
        translation_matrix = np.matrix([[1.0, 0.0, -min_x],[0.0, 1.0, -min_y],[0.0, 0.0, 1.0]])
        homography_matrix = translation_matrix * homography_matrix # * image

        # 4- save the current translation
        translation = translation_matrix * translation

        # 5- applying the transformation and blend
        composed_img = matrix_applier(composed_img, translation_matrix, (width, height))
        current_img = matrix_applier(current_img, homography_matrix, (width, height))
        composed_img = blender(composed_img, current_img)

    return composed_img

def make_stiching(x, h, base_idx=0, matrix_applier=None, blender=None, max_width=None, max_heignt=None):

    matrix_applier = matrix_applier or default_matrix_applier
    blender = blender or default_blender

    composed_img = np.zeros_like(x[base_idx])

    for img, ht in zip(x, h):

        current_img = img.copy()

        max_x, min_x, max_y, min_y = compute_dimensions(composed_img, current_img, ht)
        width = int(max_x - min_x + 1)
        height = int(max_y - min_y + 1)

        # TODO: what to cut and what to conserve
        if max_width is not None and width > max_width : width = max_width
        if max_heignt is not None and height > max_heignt : height = max_heignt

        composed_img = matrix_applier(composed_img, np.eye(3), (width, height))
        current_img = matrix_applier(current_img, ht, (width, height))
        composed_img = blender(composed_img, current_img)
    
    return composed_img
