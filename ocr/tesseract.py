# Importing all the required modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytesseract import image_to_string
from PIL import Image

class OCRTesseract:

    def __init__(self, filename, path, flag = False):
        self.myconfig = r"--psm 11 --oem 3"
        self.filename = filename
        self.path = path
        self.flag = flag
        # self.img = Image.open(self.path + self.filename)

        # Checking if any error occur during reading the images
        try:
            self.img = cv2.imread(self.path + self.filename)
            # self.img = Image.open(self.path + self.filename)
        except Exception as e:
            print(f"Error loading image: {e}")
        # self.display(self.img)

    def inverted_images(self):
        inverted_image = cv2.bitwise_not(self.img)

        if self.flag:
            cv2.imwrite(self.path + "inverted_.png", inverted_image)
            # self.display(inverted_image)
        return inverted_image

    def grayscale(self):
        gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # gray_image = self.img.convert("L")
        # pil_image = Image.fromarray(cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        # gray_image = pil_image.convert("L")

        if self.flag:
            cv2.imwrite(self.path + "gray_.png", gray_image)
            # self.display(gray_image)
        return gray_image

    def binarization(self, gray_im):
        thresh, im_bw = cv2.threshold(np.array(gray_im), 200, 230, cv2.THRESH_BINARY)

        if self.flag:
            cv2.imwrite(self.path + "binary_.png", im_bw)
            # self.display(im_bw)
        return im_bw

    def noise_removal(self, image):

        kernel = np.ones((1,1), np.uint8)
        nr_image = cv2.dilate(image, kernel, iterations = 1)
        nr_image = cv2.erode(nr_image, kernel, iterations = 1)
        nr_image = cv2.morphologyEx(nr_image, cv2.MORPH_CLOSE, kernel)
        nr_image = cv2.medianBlur(nr_image, 3)

        if self.flag:
            cv2.imwrite(self.path + "no_noise_.png", nr_image)
            # self.display(nr_image)
        return nr_image

    def thin_font(self, image):
        """   Dilation and Erosion  """
        kernel = np.ones((2,2), np.uint8)
        thin_font_im = cv2.bitwise_not(image)
        thin_font_im = cv2.erode(thin_font_im, kernel, iterations = 1)
        thin_font_im = cv2.bitwise_not(thin_font_im)

        if self.flag:
            cv2.imwrite(self.path + "thin_font_.png", thin_font_im)
            # self.display(thin_font_im)
        return thin_font_im

    def thick_font(self, image):
        """   Dilation and Erosion  """
        kernel = np.ones((2,2), np.uint8)
        thick_font_im = cv2.bitwise_not(image)
        thick_font_im = cv2.dilate(thick_font_im, kernel, iterations = 1)
        thick_font_im = cv2.bitwise_not(thick_font_im)

        if self.flag:
            cv2.imwrite(self.path + "thick_font_.png", thick_font_im)
            # self.display(thick_font_im)
        return thick_font_im

    def getSkewAngle(self, cvImage) -> float:
        # Prep image, copy, convert to gray scale, blur, and threshold
        newImage = cvImage.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations = 2)

        # Find all contours
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(newImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the largest contour and surround in min area box
        largestContour = contours[0]
        print(len(contours))
        minAreaRect = cv2.minAreaRect(largestContour)

        if self.flag:
            cv2.imwrite(self.path + "skewed_.png", newImage)

        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    @staticmethod
    def rotateImage(cvImage, angle: float):
        """ Rotate the image around its center. """
        newImage = cvImage.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
        return newImage

    def deskew(self, cvImage):
        """ Deskew image """
        angle = self.getSkewAngle(cvImage)
        rotate_im = self.rotateImage(cvImage, -1.0 * angle)

        if self.flag:
            cv2.imwrite(self.path + "rotated_fixed_.png", rotate_im)
            # self.display(rotate_im)
        return rotate_im


    def remove_borders(self, no_noise_im):
        contours, hierarchy = cv2.findContours(no_noise_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntsSorted = sorted(contours, key = lambda a:cv2.contourArea(a))
        cnt = cntsSorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = no_noise_im[y:y + h, x:x + w]

        if self.flag:
            cv2.imwrite(self.path + "no_borders_.png", crop)
            # self.display(crop)
        return crop

    def remove_footnotes(self, image):

        im_h, im_w, im_d = image.shape
        base_image = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Create rectangular structuring element and dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
        dilate = cv2.dilate(thresh, kernel, iterations = 1)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key = lambda a:cv2.boundingRect(a)[1])

        roi = -1
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h < 20 and w > 250:
                roi = base_image[0:y + h, 0:x + im_w]
                cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

        if self.flag:
            cv2.imwrite(self.path + "no_footnotes_.png" , roi)
            # self.display(roi)
        return roi

    @staticmethod
    def display(im_data):
        """ Displaying the given Image    """
        dpi = 80
        # im_data = plt.imread(im_path)
        height, width, depth = im_data.shape

        # What size does the figure need to be inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize = figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # hide spines, ticks, etc.
        ax.axis('off')

        # Display the Image
        ax.imshow(im_data, cmap = "gray")
        plt.show()

    def ocr(self, img):

        text = image_to_string(img, config = self.myconfig)
        text = text.replace("\n", " ")
        # text_file = open(self.path + "labelOCR_" + self.filename[0:-4] + ".txt", "w")
        # text_file.write(text)
        # text_file.close()
        # print("OCR extracted string from image, and stored at " + self.path + "labelOCR_" + self.filename[0:-4] + ".txt" + "...")

        return text

    def __call__(self, *args, **kwargs):

        # self.inverted_images()
        gray_im = self.grayscale()
        im_bw = self.binarization(gray_im)
        no_noise_im = self.noise_removal(im_bw)
        th_fn_im = self.thick_font(no_noise_im)
        no_border_im = self.remove_borders(th_fn_im)
        # self.deskew(self.img)
        # self.remove_footnotes(null)

        text = self.ocr(no_border_im)
        text = text.replace("\n", " ")
        return text

    def __del__(self):
        pass

