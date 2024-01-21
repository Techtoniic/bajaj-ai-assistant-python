from ocr.tesseract import OCRTesseract
from llm.gpt_turbo import DataPromptGPT
from matching.segmentation import ImageSegment
from template import TemplateMatching
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import matplotlib.image as mpimg


customer_id = random.choice(range(1,1000))
doc_type = "report"
path = "./test/" + "report.jpg"
ocr = OCRTesseract("report.jpg", "./test/")
text = ocr()
print(text)

response, jsn = DataPromptGPT().getOutput(text, category = 2)
print(jsn)


# filename = ""

# header, body, footer = ImageSegment.HBFCrop("./test/medic33.png")
tm = TemplateMatching()
stringScore = tm.contentMatch(jsn)
count = 0
filename = stringScore[0]["filename"]
for i in stringScore:
    count += 1

print(count)
x = "./test/" + doc_type + "/" + str(filename)

def dualDisplay(img1, img2):
    # Read images using Matplotlib
    image1 = mpimg.imread(img1)
    image2 = mpimg.imread(img2)

    # Create a figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display images on the axes
    axes[0].imshow(image1)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(image2)
    axes[1].set_title("Matched Image")
    axes[1].axis("off")

    # Display the result
    plt.show()

dualDisplay(path, x)