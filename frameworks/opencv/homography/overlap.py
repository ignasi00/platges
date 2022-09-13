
# Given images and homography matrix, generate masks with the overlapping region of each image with another one
# Maybe bit-wise code to index the images, number of active bits means the number of overlapping images
# Else return matrix of paired masks <-- Implemented this

import cv2
import numpy as np

from .utils import compute_path_homography, poly2mask, _image_idx_iterator


def compute_overlap(x, v_H):
    assert len(v_H) == len(x)
    
    paired_overlaps = dict()

    # The overlap is defined by the corners
    def shape_gen(x):
        for x_i in x: yield x_i.shape[:2]
    corner_coordinates_original = [((0, 0), (x_c - 1, 0), (x_c - 1, y_c - 1), (0, y_c - 1)) for x_c, y_c in shape_gen(x)]

    # the corners of the images are translated into the common plane coordinates
    def translate_corners(corner_coordinates, h):
        return cv2.perspectiveTransform(np.float32(corner_coordinates).reshape(-1,1,2), h).squeeze().round().astype(int)

    corner_coordinates_common = [translate_corners(corner_coordinates, h) for corner_coordinates, h in zip(corner_coordinates_original, v_H)]

    for base_idx, h in enumerate(v_H):
        max_x, max_y = corner_coordinates_original[base_idx][2]
        # One image is fully overlapped with itself
        paired_overlaps.update({ (base_idx, base_idx) : np.ones( (max_x + 1, max_y + 1) ) })

        h_inv = np.linalg.inv(h)
        
        for i in range(len(x)):
            if i == base_idx : continue

            i_on_common_coordiantes = corner_coordinates_common[i].copy()
            i_on_base_coordinates = translate_corners(i_on_common_coordiantes, h_inv)

            # and clipped in the base image shape
            i_on_base_coordinates[ i_on_base_coordinates < 0] = 0
            i_on_base_coordinates[ i_on_base_coordinates[:, 0] > max_x , 0] = max_x
            i_on_base_coordinates[ i_on_base_coordinates[:, 1] > max_y , 1] = max_y

            # this polygon defines the overlap (a binary mask is stored instead of the polygon)
            paired_overlaps.update({ (base_idx, i) : poly2mask(i_on_base_coordinates.flatten().tolist(), max_x + 1, max_y + 1) })

    return paired_overlaps

def compute_overlap_from_paired_homographies(x, H, n_matches):
    # TODO: verbose option -> tqdm like fors
    paired_overlaps = dict()

    # The overlap is defined by the corners
    def shape_gen(x):
        for x_i in x: yield x_i.shape
    corner_coordinates_original = [((0, 0), (x_c - 1, 0), (x_c - 1, y_c - 1), (0, y_c - 1)) for x_c, y_c in shape_gen(x)]

    for base_idx in range(len(x)):
        max_x, max_y = corner_coordinates_original[base_idx][2]
        # One image is fully overlapped with itself
        paired_overlaps.update({ (base_idx, base_idx) : np.ones( (max_x + 1, max_y + 1) ) })

        # Each image has a position on the base image plane that can be used to compute the position of its neighbour images, _image_idx_iterator define the neigbourhood path.
        for current_idxs in _image_idx_iterator(n_matches, base_idx):
            i = base_idx
            j = current_idxs[-1]

            # Given the neighbours of the current image and the paired homographies, compute the homography for the current image pixels to reach the base image plane.
            h = compute_path_homography(current_idxs, H)

            # If no homagraphy is found, it can be assumed that no overlap exist
            if h is None:
                paired_overlaps.update({ (i, j) : np.zeros( (max_x + 1, max_y + 1) ) })

            # When a homography matrix is found,
            else:
                # the corners of the current image are translated into the base image plane
                j_on_i_coordiantes = cv2.perspectiveTransform(np.float32(corner_coordinates_original[j]).reshape(-1,1,2), h)
                j_on_i_coordiantes = np.squeeze(j_on_i_coordiantes) # Remove extra dimensions
                j_on_i_coordiantes = np.round(j_on_i_coordiantes).astype(int)   # Convert to integer

                # and clipped in the base image shape
                j_on_i_coordiantes[ j_on_i_coordiantes < 0] = 0
                j_on_i_coordiantes[ j_on_i_coordiantes[:, 0] > max_x , 0] = max_x
                j_on_i_coordiantes[ j_on_i_coordiantes[:, 1] > max_y , 1] = max_y

                # this polygon defines the overlap (a binary mask is stored instead of the polygon)
                paired_overlaps.update({ (i, j) : poly2mask(j_on_i_coordiantes.flatten().tolist(), max_x + 1, max_y + 1) })
    
    return paired_overlaps

