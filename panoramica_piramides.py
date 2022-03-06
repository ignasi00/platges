
import cv2
from matplotlib import pyplot as plt

from other_algorithms.overlap_finder import BatchOverlapFinder, RANSAC_MatrixFinder


x = [f'/mnt/c/Users/Ignasi/Downloads/piramides/cairo{i}.png' for i in range(1, 4)]

images = []
pathes = []
for path in x:
    images.append(cv2.imread(path, 0))
    pathes.append(path)

bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
matrix_finder = RANSAC_MatrixFinder(point_matcher=bf,
                                    MIN_MATCH_COUNT=4,
                                    ransacReprojThreshold=5,
                                    maxIters=1000)

bof = BatchOverlapFinder(homography_matrix_estimator=matrix_finder)

bof.train()
pano, n_matches = bof(images)

plt.imshow(pano)
plt.show()

bof.eval()
overlaps, n_matches = bof(images)

plt.imshow(overlaps[0, 1])
plt.show()
