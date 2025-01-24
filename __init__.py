from .face_crop import FaceCrop

NODE_CLASS_MAPPINGS = {
    "Face Crop": FaceCrop,
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS"]
