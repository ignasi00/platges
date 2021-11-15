
import cv2
from torch.utils.data import DataLoader

from datasets.platges_datasets import Platges_DronHomographyDataset
from models.algorithms.homography import BasicStitching, RANSAC_MatrixFinder


DATA_PATH = '/mnt/c/Users/Ignasi/Downloads/4fotos_ret' #'/mnt/c/Users/Ignasi/Downloads/PlatgesBCN/dron'
SAVE_PATH = '/mnt/c/Users/Ignasi/Downloads/panorama_test.jpg'


HANDPICKED_KP = (
    (
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1)
    ),
    (
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1)
    ),
    (
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1)
    ),
    (
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1),
        cv2.KeyPoint(, , 1)
    )
)

# HANDPICKED_KP = None


if __name__ == "__main__":
    
    dataset = Platges_DronHomographyDataset(DATA_PATH, 
                                            to_tensor=False, 
                                            cluster=True,#True, 
                                            th_time=60, 
                                            longitude_bin_size=180, 
                                            latitude_bin_size=180, 
                                            min_per_bin=1,
                                            downsample=4)#4)
    def my_collate(x): return x # <- do not transform imgs to tensor
    dataloader = DataLoader(dataset, collate_fn=my_collate)

    # TODO: mask beach from cityscapes

    descriptor = cv2.SIFT_create()
    point_finder = lambda img, kp : descriptor.compute(img, kp)
    # point_finder = None

    bf = cv2.BFMatcher(crossCheck=True) # Brute Force Matcher
    matrix_finder = RANSAC_MatrixFinder(point_matcher=bf,
                                        MIN_MATCH_COUNT=4,
                                        ransacReprojThreshold=10,
                                        maxIters=1000)
    sticher = BasicStitching(point_finder_descriptor=point_finder, homography_matrix_estimator=matrix_finder)

    for batch in dataloader:
        for imgs, metas in batch:
            print('\n'.join([f"{i} - {meta['path']}" for i, meta in enumerate(metas)]))
            print("\n---\n")

            panorama = sticher(imgs[0:], verbose=True, plot=False, descriptor_arg2=HANDPICKED_KP)

            cv2.imwrite(SAVE_PATH, panorama)

            raise
