# Importing all the required modules
from PIL import Image

class ImageSegment:

    def __init__(self):
        pass

    @staticmethod
    def HBFCrop(image_path, header_percentage = 18, footer_percentage = 15):
        # Open the image
        image = Image.open(image_path)

        # Get image dimensions
        width, height = image.size

        # Calculate crop positions
        header_height = int(height * (header_percentage / 100))
        footer_height = int(height * (footer_percentage / 100))

        # Crop header
        header_box = (0, 0, width, header_height)
        header = image.crop(header_box)

        # Crop footer
        footer_box = (0, height - footer_height, width, height)
        footer = image.crop(footer_box)

        # Crop body
        body_box = (0, header_height, width, height - footer_height)
        body = image.crop(body_box)

        # Save cropped images
        # header.save('header.jpg')
        # body.save('body.jpg')
        # footer.save('footer.jpg')

        return header, body, footer

    @staticmethod
    def HBCrop(image_path, header_percentage = 18):
        # Open the image
        image = Image.open(image_path)

        # Get image dimensions
        width, height = image.size

        # Calculate the crop height for the header (top 18%)
        crop_height_header = int(0.18 * height)

        # Crop the header
        header = image.crop((0, 0, width, crop_height_header))

        # Crop the body (rest of the image)
        body = image.crop((0, crop_height_header, width, height))

        # Save cropped images
        # header.save('header.jpg')
        # body.save('body.jpg')

        return header, body
