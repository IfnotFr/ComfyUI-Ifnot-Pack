from .face_crop import FaceCrop
from .face_crop_mouth import FaceCropMouth

NODE_CLASS_MAPPINGS = {
    "Face Crop": FaceCrop,
    "Face Crop Mouth": FaceCropMouth,
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS"]
