import os
import cv2
import io
import base64
from PIL import Image


def write_file_to_temp_folder(file, temp_dir):
    """
    Writes a Streamlit File object to the session temporary folder, and returns the filepath.
    """
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.read())
    return path


def load_image(image_path: str, use_cv2: bool = False):
    if use_cv2:
        return cv2.imread(image_path)
    return Image.open(image_path)


def encode_image_base64(image):
    """
    Takes a PIL image, converts it to bytes, and encodes it with base 64.
    """
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="png") #image.format)
    img_byte_array = img_byte_array.getvalue()
    return base64.b64encode(img_byte_array).decode("utf-8")
