import io
import cv2
import os
import imagehash
import numpy as np
import chromadb
from chromadb.config import Settings
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input

# Load the VGG16 model pre-trained on ImageNet
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract embeddings from an image
def image_to_embeddings(image_path):
    img_array = load_and_preprocess_image(image_path)
    embeddings = model.predict(img_array)
    embeddings = embeddings.flatten()  # Flatten the embeddings to a 1D vector
    return embeddings

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

    # Define the lower and upper bounds of the color you want to segment
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
    #print(f"Similarity Score: {similarity_score}")
    #print(f"Percentage Similarity: {percentage_similarity:.2f}%")

    plt.imshow(matching_image)
    plt.axis("off")
    plt.show()
    return percentage_similarity

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

        """plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(image2_with_box, cv2.COLOR_BGR2RGB))
        plt.title('Matching Objects (with Box)')
        plt.axis('off')"""

        plt.show()
        return percentage_similarity
    else:

        print("Not enough matches found.")
        return 0

"""
def db():
    client = chromadb.PersistentClient(path="./")
    client.heartbeat()
    settings = Settings(chroma_api_impl="chromadb.api.fastapi.FastAPI")
    chroma_client = chromadb.HttpClient(host="localhost", port=8000, settings=settings)

    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    # collection = client.create_collection("all-new-documents")
    collection = client.get_collection(name="all-my-documents")

    return collection

def add_data(collection, x, y, z):
    # Add docs to the collection. Can also update and delete. Row-based API coming soon!
    collection.add(
        documents=[str(x),str(y)],     # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
        ids=["id1","id2"]  # unique for each doc
    )


dbt = db()
count = 1
for i in os.listdir('./test/vector/'):
    imageEmbeddings = image_to_embeddings('./test/vector/' + i)
    # print(imageEmbeddings)
    add_data(dbt, i, imageEmbeddings, count)
    count = count + 1

client = chromadb.HttpClient(host="localhost", port=8000)
cld = client.get_collection(name="all-new-documents")
print(cld.get())"""
# print(cld, cld[0])
# compare_images('./test/medic22.jpg', './test/medic33.png')
# compare_images_shifted_objects('./test/medic22.jpg', './test/medic33.png')

image_path = './test/SIH_eval/aks_sample.jpeg'

max_score = 0
for i in os.listdir("./test/SIH_eval/"):
    for j in os.listdir("./test/SIH_eval"):
        if i != j:
            # print(i, j)
            score = compare_images_shifted_objects("./test/SIH_eval/"+i, './test/SIH_eval/'+j)
            if score >= 30:
                print(score, i, j)
            if score > max_score:
                max_score = score
print(max_score)
# print("Image embeddings shape:", image_embeddings.shape)
# print(compare_images_shifted_objects("./test/SIH_eval/aks_Sample.jpeg", "./test/SIH_eval/sag_sample.jpeg"))
# print(compare_images("it status
# ./test/SIH_eval/aks_Sample.jpeg", "./test/SIH_eval/sag_sample.jpeg"))