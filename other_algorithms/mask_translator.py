
import cv2
import numpy as np

from .overlap_finder import RANSAC_MatrixFinder


class MaskTranslator():

    def _compute_dimensions(self, composed_img, current_img, homography_matrix):
        h, w = composed_img.shape
        composed_corners = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2).squeeze().astype(int)

        h, w = current_img.shape
        pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        next_corners = cv2.perspectiveTransform(pts, homography_matrix)
        next_corners = np.squeeze(next_corners) # Remove extra dimensions
        next_corners = np.round(next_corners).astype(int)   # Convert to integer

        x_coord = np.concatenate([composed_corners, next_corners]).squeeze()[:,0]
        y_coord =  np.concatenate([composed_corners, next_corners]).squeeze()[:,1]

        max_x = x_coord.max()
        min_x = x_coord.min()
        max_y = y_coord.max()
        min_y = y_coord.min()

        return max_x, min_x, max_y, min_y

    def __init__(self, point_finder_descriptor=None, homography_matrix_estimator=None, matrix_applier=None, training=True):

        self.training = training

        self.descriptor = point_finder_descriptor
        if self.descriptor is None:
            # points of interest with histogram of gradient not direction dependent descriptors
            descriptor = cv2.SIFT_create()
            self.descriptor = lambda img, other : descriptor.detectAndCompute(img, other)
    
        self.matrix_finder = homography_matrix_estimator
        if self.matrix_finder is None:
            bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
            self.matrix_finder = RANSAC_MatrixFinder(point_matcher=bf, MIN_MATCH_COUNT=4, ransacReprojThreshold=5.0, maxIters=5)

        self.matrix_applier = matrix_applier
        if self.matrix_applier is None:
            self.matrix_applier = lambda img, h, size : cv2.warpPerspective(img, h, size, flags=cv2.INTER_LINEAR)

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
        
    def forward(self, img1, img2, mask1, mask2):
        kp1, des1 = self.descriptor(img1, None)
        kp2, des2 = self.descriptor(img2, None)

        H, _, mask = self.matrix_finder(kp1, des1, kp2, des2)

        if self.training:
            max_x, min_x, max_y, min_y = self._compute_dimensions(img1, img2, H)
            width = int(max_x - min_x + 1)
            height = int(max_y - min_y + 1)

            translation_matrix = np.matrix([[1.0, 0.0, -min_x],[0.0, 1.0, -min_y],[0.0, 0.0, 1.0]])
            h = translation_matrix * H # * image

            msk1_h = cv2.warpPerspective(mask1, translation_matrix, (width, height), flags=cv2.INTER_LINEAR) # mask1 translated
            msk2_h = cv2.warpPerspective(mask2, h, (width, height), flags=cv2.INTER_LINEAR) #mask2 homography into 1 translated

            img1_h = cv2.warpPerspective(img1, translation_matrix, (width, height), flags=cv2.INTER_LINEAR)
            img2_h = cv2.warpPerspective(img2, h, (width, height), flags=cv2.INTER_LINEAR)

            return msk1_h, msk2_h, img1_h, img2_h
        else:
            msk1_h = cv2.warpPerspective(mask1, np.linalg.inv(H), (width, height), flags=cv2.INTER_LINEAR) # mask1 into 2
            msk2_h = cv2.warpPerspective(mask2, H, (width, height), flags=cv2.INTER_LINEAR) # mask2 into 1
            return msk1_h, msk2_h

    __call__ = forward
