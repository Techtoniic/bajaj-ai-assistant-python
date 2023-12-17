# Importing all the required modules
import cv2

class SimilarityCalc:

    def __init__(self):
        pass

    @staticmethod
    def sift(img1, img2):

        # Load your images
        # img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        # img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        count = 0
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

            count += 1

        return (len(good_matches) / count) * 100

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
