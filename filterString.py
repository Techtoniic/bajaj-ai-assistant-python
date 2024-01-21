from ocr.tesseract import OCRTesseract
import random
from llm.gpt_turbo import DataPromptGPT
import io
import cv2
import imagehash
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim


def remove_keywords(input_string, keywords):
    # Split the input string into words
    words = input_string.split()

    # Remove keywords from the list of words
    if keywords:
        filtered_words = [word for word in words if word.lower() not in map(lambda x: str(x).lower() if x else "", keywords)]
    else:
        filtered_words = words

    # Join the remaining words to form the modified string
    result_string = ' '.join(filtered_words)

    return result_string


def levenshtein_distance(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    # Create a matrix to store the edit distances
    matrix = [[0] * len_str2 for _ in range(len_str1)]

    # Initialize the matrix
    for i in range(len_str1):
        matrix[i][0] = i
    for j in range(len_str2):
        matrix[0][j] = j

    # Fill in the matrix
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # deletion
                matrix[i][j - 1] + 1,  # insertion
                matrix[i - 1][j - 1] + cost  # substitution
            )

    # The final value in the matrix represents the Levenshtein distance
    levenshtein_distance = matrix[-1][-1]

    # Calculate similarity percentage
    max_length = max(len(str1), len(str2))
    similarity_percentage = ((max_length - levenshtein_distance) / max_length) * 100

    return similarity_percentage


def compare_images_ssim(image_path1, image_path2):
    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute structural similarity index
    similarity_index, _ = ssim(image1, image2, full=True)

    return similarity_index



def compare_images(image_path1, image_path2):
    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert BGR to HSV for color-based segmentation
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the color you want to segment (e.g., green)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([255, 255, 255])

    # Threshold the images to get only the specified color range
    mask1 = cv2.inRange(hsv_image1, lower_bound, upper_bound)
    mask2 = cv2.inRange(hsv_image2, lower_bound, upper_bound)

    # Bitwise-AND mask and original images
    segmented_image1 = cv2.bitwise_and(image1, image1, mask=mask1)
    segmented_image2 = cv2.bitwise_and(image2, image2, mask=mask2)

    # Use ORB for feature matching
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(segmented_image1, None)
    kp2, des2 = orb.detectAndCompute(segmented_image2, None)

    # Create BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches on the segmented images
    matching_image = cv2.drawMatches(segmented_image1, kp1, segmented_image2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    # cv2.imshow('Matching Result', matching_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Calculate similarity score
    similarity_score = len(matches)

    # Calculate percentage similarity based on the total number of keypoints
    total_keypoints = max(len(kp1), len(kp2))
    percentage_similarity = (similarity_score / total_keypoints) * 100

    # Print and return the results
    print(f"Similarity Score: {similarity_score}")
    print(f"Percentage Similarity: {percentage_similarity:.2f}%")

    plt.imshow(matching_image)
    plt.axis("off")
    plt.show()

def compare_images_shifted_objects(image_path1, image_path2):
    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Create BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches on the images
    matching_image = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Use RANSAC to find homography
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = gray1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # Transform the corners using the homography matrix
        dst = cv2.perspectiveTransform(pts, H)

        # Draw the bounding box on the second image
        image2_with_box = cv2.polylines(image2, [np.int32(dst)], True, (0, 255, 0), 3)

        # Print and return the results
        print(f"Number of Good Matches: {len(good_matches)}")

        # Calculate similarity score
        similarity_score = len(matches)

        # Calculate percentage similarity based on the total number of keypoints
        total_keypoints = max(len(kp1), len(kp2))
        percentage_similarity = (similarity_score / total_keypoints) * 100

        # Print and return the results
        print(f"Similarity Score: {similarity_score}")
        print(f"Percentage Similarity: {percentage_similarity:.2f}%")


        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        plt.title('Image 1')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        plt.title('Image 2')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(image2_with_box, cv2.COLOR_BGR2RGB))
        plt.title('Matching Objects (with Box)')
        plt.axis('off')

        plt.show()
    else:
        print("Not enough matches found.")







customer_id = random.choice(range(1,1000))
doc_type = "invoice"

path1 = "./test/" + "inovo.jpg"
ocr = OCRTesseract("inovo.jpg", "./test/")
text1 = ocr()

path2 = "./test/" + "report.jpg"
ocr = OCRTesseract("report.jpg", "./test/")
text2 = ocr()
# print(text1,"\n", text2)

response1, jsn1 = DataPromptGPT().getOutput(text1, category = 1)
response2, jsn2 = DataPromptGPT().getOutput(text2, category = 2)
# print(jsn1, "\n", jsn2)

res1 = remove_keywords(text1, jsn1)
res2 = remove_keywords(text2, jsn2)

print(res1, "\n", res2)

similarity_percentage = levenshtein_distance(res1, res2)

print(f"Similarity Percentage: {similarity_percentage}%")

"""similarity_score = score(distance(res1, res2))
# similarity_score = score("Avalanche", "Apocalypse")
print(f"Similarity Score: {similarity_score:.2f}%")

similarity_percentage = calculate_cosine_similarity(res1, res2)
print("Cosine Similarity Percentage:", similarity_percentage)"""