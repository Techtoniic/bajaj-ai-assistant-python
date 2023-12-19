# Importing all the required modules
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pysiftn import computeKeypointsAndDescriptors
from skimage.metrics import structural_similarity

class SimilarityCalc:

    def __init__(self):
        pass

    @staticmethod
    def sift(img1, img2):

        # Load your images
        # img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        MIN_MATCH_COUNT = 10
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # kp1, des1 = computeKeypointsAndDescriptors(img1)
        # kp2, des2 = computeKeypointsAndDescriptors(img2)

        count = 0
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

            count += 1

        return (len(good_matches) / len(kp2)) * 100, len(good_matches)

        """# Initialize and use FLANN
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            # Estimate homography between template and scene
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

            # Draw detected template in scene image
            h, w = img1.shape
            pts = np.float32([[0, 0],
                              [0, h - 1],
                              [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            h1, w1 = img1.shape
            h2, w2 = img2.shape
            nWidth = w1 + w2
            nHeight = max(h1, h2)
            hdif = int((h2 - h1) / 2)
            newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

            for i in range(3):
                newimg[hdif:hdif + h1, :w1, i] = img1
                newimg[:h2, w1:w1 + w2, i] = img2

            # Draw SIFT keypoint matches
            for m in good:
                pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
                pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
                cv2.line(newimg, pt1, pt2, (255, 0, 0))

            plt.imshow(newimg)
            plt.show()
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            return MIN_MATCH_COUNT"""


    @staticmethod
    def ssd(img1, img2):

        # Load your images
        # img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

        result = cv2.matchTemplate(img1, img2, cv2.TM_SQDIFF_NORMED)

        if result is not None:
            _, max_similarity, _, _ = cv2.minMaxLoc(result)
            return max_similarity * 100
        else:
            return None


    @staticmethod
    def ncc(img1, img2):

        # Load your images
        # img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)

        if result is not None:
            _, max_similarity, _, _ = cv2.minMaxLoc(result)
            return max_similarity * 100
        else:
            return None

    @staticmethod
    def ssim(img1, img2):
        score, diff = structural_similarity(img1.reshape(1, -1), img2.reshape(1, -1), data_range=1, full=True,
                                            gaussian_weights=False, use_sample_covariance=False)
        return score * 100