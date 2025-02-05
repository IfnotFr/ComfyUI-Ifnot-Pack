import torch
import mediapipe as mp
import numpy as np
import cv2


class GetBeardMask:
    CATEGORY = "Ifnot Pack"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MASK", "MASK")
    FUNCTION = "get_beard_mask"

    def get_beard_mask(
        self,
        images,
        extra_pnginfo=None,
    ):
        """
        Main entry point of the node.
        """

        # -- Load face mesh and prepare image
        image = self._torch_to_numpy(images[0])
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # -- Detect the landmarks of the face in the image
        landmarks = self._get_face_landmarks(face_mesh, image)
        if landmarks is None:
            return (image.unsqueeze(0), image.unsqueeze(0))

        # -- Based on the landmarks, generate the mouth mask
        mouth_mask = self._create_mouth_mask(image, landmarks)

        # -- Based on the landmarks, generated a wide bottom face mask for the beard
        beard_mask = self._create_beard_mask(image, landmarks)

        # Return the torch masks as torch tensors tuples
        return (
            self._numpy_to_torch(beard_mask).unsqueeze(0),
            self._numpy_to_torch(mouth_mask).unsqueeze(0),
        )

    def _torch_to_numpy(self, image_torch):
        """Convert a Torch image [C, H, W] to a NumPy RGB image [H, W, C]."""
        return (image_torch.cpu().numpy() * 255).astype("uint8")

    def _numpy_to_torch(self, image):
        """Convert to a normalized torch tensor [C, H, W]."""
        return torch.from_numpy(image.astype("float32") / 255.0)

    def _get_face_landmarks(self, face_mesh, image):
        """Return the landmarks of the first face detected in the image."""
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None
        """
        Return the bounding box of the face.
        """
        x_coords = [lm.x * width for lm in landmarks]
        y_coords = [lm.y * height for lm in landmarks]

        x_min = int(max(0, min(x_coords)))
        x_max = int(min(width, max(x_coords)))
        y_min = int(max(0, min(y_coords)))
        y_max = int(min(height, max(y_coords)))
        return x_min, y_min, x_max, y_max

    def _create_mouth_mask(self, image, landmarks):
        """
        Create a mask (H, W) of the mouth based on the landmarks
        detected in the aligned image. The outer lip indices are used.
        """
        height, width = image.shape[:2]

        # Indices of the outer lip contour (source Mediapipe)
        # The inner arc can be included if we want the whole mouth region.
        MOUTH_OUTLINE = [
            0,
            267,
            269,
            270,
            409,
            306,
            375,
            321,
            405,
            314,
            17,
            84,
            181,
            91,
            146,
            61,
            185,
            40,
            39,
            37,
        ]

        # Get all (x, y) points in pixels
        points = []
        for idx in MOUTH_OUTLINE:
            x = int(landmarks[idx].x * width)
            y = int(landmarks[idx].y * height)
            points.append([x, y])

        # Convert to NumPy array for fillPoly
        points = np.array([points], dtype=np.int32)

        # Create a black mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the mouth in white
        cv2.fillPoly(mask, points, 255)

        return mask

    def _create_beard_mask(self, image, landmarks):
        """
        Create a mask (H, W) of the bottom of the head based on the landmarks
        detected in the aligned image. The outer lip indices are used.
        """
        height, width = image.shape[:2]

        # Indices of the bottom of the face
        BEARD_OUTLINE = [
            # Top
            227,
            123,
            205,
            203,
            2,
            423,
            425,
            280,
            447,
            
            # Bottom
            454,
            323,
            361,
            288,

            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,

            58,
            132,
            93,
            234,
            
        ]

        ADD_Y = {
            454: 0,
            323: 2,
            361: 2,
            288: 5,

            397: 10,
            365: 10,
            379: 10,
            378: 15,
            400: 20,
            377: 20,
            152: 20,
            148: 20,
            176: 20,
            149: 15,
            150: 10,
            136: 10,
            172: 10,

            58: 5,
            132: 2,
            93: 2,
            234: 0,
        }

        # Get all (x, y) points in pixels
        points = []
        for idx in BEARD_OUTLINE:
            x = int(landmarks[idx].x * width)
            y = int(landmarks[idx].y * height + ADD_Y.get(idx, 0))
            points.append([x, y])

        # Convert to NumPy array for fillPoly
        points = np.array([points], dtype=np.int32)

        # Create a black mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the form of the beard in white
        cv2.fillPoly(mask, points, 255)

        return mask
