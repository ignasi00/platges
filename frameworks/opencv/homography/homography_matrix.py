
import cv2
import numpy as np

from .utils import _image_idx_iterator


def _default_point_finder():
    # kp = point_finder(ima, mask=None)
    point_finder_obj = cv2.SIFT_create()
    point_finder = lambda img, mask : point_finder_obj.detect(img, mask)
    return point_finder

def _default_descriptor():
    # des, kp = descriptor_finder(ima, kp)
    descriptor_finder_obj = cv2.SIFT_create()
    descriptor_finder = lambda img, kp : descriptor_finder_obj.compute(img, kp)
    return descriptor_finder

def _default_matrix_finder():
    # H12, matches, inliers_mask = matrix_finder(kp1, des1, kp2, des2) # if no H found, H == None
    bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
    point_matcher = lambda des1, des2 : bf.match(des1, des2)
    matrix_finder = RANSAC_MatrixEstimator(point_matcher=point_matcher, MIN_MATCH_COUNT=4, ransacReprojThreshold=5.0, maxIters=5)
    return matrix_finder

def compute_homography(img1, img2, mask1=None, mask2=None, point_finder=None, descriptor_finder=None, matrix_finder=None):
    kp1       = point_finder(img1, mask=mask1)
    des1, kp1 = descriptor_finder(img1, kp1)

    kp2       = point_finder(img2, mask=mask2)
    des2, kp2 = descriptor_finder(img2, kp2)

    H12, matches, inliers_mask = matrix_finder(kp1, des1, kp2, des2)
    return H12, matches, inliers_mask

##################################################

class HomographyMatrixEstimator():

    def __init__(self, point_finder=None, descriptor_finder=None, matrix_finder=None):
        
        self.point_finder = point_finder or _default_point_finder()
        self.descriptor_finder = descriptor_finder or _default_descriptor_finder()
        self.matrix_finder = matrix_finder or _default_matrix_finder()
    
    def forward(self, img1, img2, mask1=None, mask2=None):
        H12, matches, inliers_mask = compute_homography(img1, img2, mask1=mask1, mask2=mask2, point_finder=self.point_finder, descriptor_finder=self.descriptor_finder, matrix_finder=self.matrix_finder)
        return H12, matches, inliers_mask, sum(inliers_mask)

    __call__ = forward

class BatchPairedHomographyMatrixEstimator():
    # Compute all the paired images homography matrixs

    def __init__(self, point_finder=None, descriptor_finder=None, matrix_finder=None):
        
        self.point_finder = point_finder or _default_point_finder()
        self.descriptor_finder = descriptor_finder or _default_descriptor_finder()
        self.matrix_finder = matrix_finder or _default_matrix_finder()
    
    def forward(self, imgs, masks=None):
        if masks == None:
            masks = [None] * len(imgs)

        assert (len(imgs) == len(masks)) # or (masks == None)

        def detect_and_compute(img, mask):
            kp      = self.point_finder(img, mask=mask)
            des, kp = self.descriptor_finder(img, kp)
            return kp, des
        
        v_kp, v_des = zip(*[detect_and_compute(img, mask) for img, mask in zip(imgs, masks)])

        H = dict()
        matches = dict()
        inliers_mask_all = dict()
        n_matches = np.zeros((len(v_kp), len(v_kp)))

        for idx1, (kp1, des1) in enumerate(zip(v_kp, v_des)):
            H.update( {(idx1, idx1) : np.eye(3)} )

            for idx2, (kp2, des2) in enumerate(zip(v_kp[idx1 + 1:], v_des[idx1 + 1:])):
                h, img_matches, inliers_mask = self.matrix_finder(kp1, des1, kp2, des2)

                H.update( {(idx1, idx2 + idx1 + 1) : np.matrix(h)} )
                matches.update( {(idx1, idx2 + idx1 + 1) : img_matches} )
                inliers_mask_all.update( {(idx1, idx2 + idx1 + 1) : inliers_mask} )
                try:
                    H.update( {(idx2 + idx1 + 1, idx1) : np.linalg.inv(np.matrix(h))} )
                except:
                    H.update( {(idx2 + idx1 + 1, idx1) : None} )
                try:
                    n_matches[idx1, idx2 + idx1 + 1] = sum(inliers_mask)
                except:
                    n_matches[idx1, idx2 + idx1 + 1] = 0
        
        n_matches = n_matches + n_matches[::-1, ::-1]
        return H, matches, inliers_mask_all, n_matches

    __call__ = forward

class BatchHomographyMatrixEstimator():
    def __init__(self, point_finder=None, descriptor_finder=None, matrix_finder=None):
        self.homographies_computator = BatchPairedHomographyMatrixEstimator(point_finder=point_finder, descriptor_finder=descriptor_finder, matrix_finder=matrix_finder)

    def forward(self, imgs, masks=None, base_idx=0):
        homographies = [None] * len(imgs)
        H, matches, inliers_mask, n_matches = self.homographies_computator(imgs, masks)

        translation = np.eye(3, 3)
        for current_idxs in _image_idx_iterator(n_matches, base_idx):

            homography_matrix = compute_path_homography(current_idxs, H)
            if homography_matrix is None:
                continue
            homography_matrix = translation * homography_matrix

            max_x, min_x, max_y, min_y = compute_dimensions(composed_img, current_img, homography_matrix)
            translation_matrix = np.matrix([[1.0, 0.0, -min_x],[0.0, 1.0, -min_y],[0.0, 0.0, 1.0]])

            homography_matrix = translation_matrix * homography_matrix # * image

            homographies[current_idxs[-1]] = homography_matrix
            translation = translation_matrix * translation
        
        # [(h is None) for h in homographies] <-- Mask of images with an asociated homography
        return homographies, matches, inliers_mask, n_matches
    
    __call__ = forward

##################################################

class Dense_PointFinder():
    # Given an image and masks, put N points as a grid
    def __init__(self, npoints=100):
        self.npoints = npoints

    def __call__(self, img, mask=None):
        if mask == None : mask = np.ones(img.shape[:2])

        coordinates = np.argwhere(mask != 0)
        coordinates = coordinates[np.int16(np.linspace(0, len(coordinates) - 1, self.npoints))]
        
        return cv2.KeyPoint_convert(coordinates.tolist())

class UnmatchedKeypoints_PointFinder():
    # Images are described by an ID function, and a dictionary that relate id with keypoints list is used
    def __init__(self, id_funct, id_keypoints_dict):
        self.id_funct = id_funct
        self.id_keypoints_dict = id_keypoints_dict

    def __call__(self, img, mask):
        id_ = self.id_funct(img)
        keypoints = self.id_keypoints_dict[id_]

        if mask is not None:
            np_keypoints = np.asarray(keypoints).T
            keypoints = keypoints[mask[np_keypoints[0], np_keypoints[1]] == 1]
            return cv2.KeyPoint_convert(keypoints)
        return keypoints

class UnmatchedDescriptors_Descriptor():
    # Images are described by an ID function, and a dictionary that relate id with keypoints list is used
    def __init__(self, id_funct, id_descriptors_dict):
        self.id_funct = id_funct
        self.id_descriptors_dict = id_descriptors_dict

    def __call__(self, img, kp):
        id_ = self.id_funct(img)
        descriptors = self.id_descriptors_dict[id_]

        assert (len(kp) == len(descriptors)), "masked manual descriptors are not supported"

        return descriptors, kp
    
class RANSAC_MatrixEstimator():
    def __init__(self, point_matcher, min_match_count=4, ransacReprojThreshold=5.0, maxIters=5):
        # ransacReprojThreshold is inliear/outlier th, If kp1, kp2 measured in pixels: th usually somewhere between 1 and 10.
        self.point_matcher = point_matcher
        self.min_match_count = min_match_count
        self.ransacReprojThreshold = ransacReprojThreshold
        self.maxIters = maxIters

    def __call__(self, kp1, des1, kp2, des2):
        # kp is key_points; des is descriptors

        # least mean squeres matching of descriptors
        matches = self.point_matcher(des1, des2)

        # RANSAC iterative adjust of matches
        if len(matches) > self.min_match_count:

            # Given the LMS matched descriptors, make matched keypoints (src_pts12[0] is only matched with dst_pts12[0])
            src_pts12 = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
            dst_pts12 = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
            
            # Estimate the homography matrix using the matched keypoints, discard outliers using RANSAC
            H, mask = cv2.findHomography(src_pts12, dst_pts12, cv2.RANSAC, self.ransacReprojThreshold, maxIters=self.maxIters)
            
            return H, matches, mask
        return None, matches, None

def MatchedKeypoints_RANSAC_MatrixEstimator(min_match_count=4, ransacReprojThreshold=5.0, maxIters=5):
    # It assume kp1 and kp2 are already matched keypoints (same idx is a correspondance)
    # des1 and des2 can be whatever object whose len(object) == len(kp)

    point_matcher = lambda des1, des2 : [cv2.DMatch(i, i, 0) for i in range(len(des1))]
    matrixEstimator = RANSAC_MatrixEstimator(point_matcher=point_matcher, min_match_count=min_match_count, ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters)
    
    return matrixEstimator
