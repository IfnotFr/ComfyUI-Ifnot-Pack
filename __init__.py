from .face_crop import FaceCrop
from .face_crop_mouth import FaceCropMouth
from .get_beard_mask import GetBeardMask

NODE_CLASS_MAPPINGS = {
    "Face Crop": FaceCrop,
    "Face Crop Mouth": FaceCropMouth,
    "Get Beard Mask": GetBeardMask,
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS"]
