
from PIL import Image

def repack_image(image):
    drawn_image = image['composite']
    image = drawn_image.convert("P")
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    return image
