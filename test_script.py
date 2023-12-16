
"""from PIL import Image, ExifTags
img = Image.open("./test/" + "edited.jpg")
img_exif = img.getexif()
print(type(img_exif))
# <class 'PIL.Image.Exif'>

if img_exif is None:
    print('Sorry, image has no exif data.')
else:
    for key, val in img_exif.items():
        if key in ExifTags.TAGS:
            print(f'{ExifTags.TAGS[key]}:{val}')
        else:
            print(f'{key}:{val}')

from exif import Image
with open("./test/" + "edited.jpg", 'rb') as image_file:
    my_image = Image(image_file)
print(my_image.has_exif)
print(my_image.list_all())"""

from invoice2data import extract_data
result = extract_data('./test/Payment Acknowledgement_INV231127000101.pdf')