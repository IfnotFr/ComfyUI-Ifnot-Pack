import torch
import mediapipe as mp
import numpy as np
import cv2


class FaceCropMouth:
    CATEGORY = "Ifnot Pack"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_images": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "move_images_1": ("IMAGE",),
                "move_images_2": ("IMAGE",),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "face_crop_mouth"

    def face_crop_mouth(
        self,
        crop_images,
        reference_images,
        move_images_1,
        move_images_2,
        extra_pnginfo=None,
    ):
        """
        Main entry point of the node.
        """
        crop_image = self._torch_to_numpy(crop_images[0])
        reference_image = self._torch_to_numpy(reference_images[0])
        move_image_1 = self._torch_to_numpy(move_images_1[0])
        move_image_2 = self._torch_to_numpy(move_images_2[0])

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # image2 = self._convert_to_torch(image)
        # return (image2.unsqueeze(0),)

        reference_landmarks = self._get_face_landmarks(face_mesh, reference_image)
        crop_landmarks = self._get_face_landmarks(face_mesh, crop_image)

        if reference_landmarks is None or crop_landmarks is None:
            return (crop_image.unsqueeze(0),)

        # -- Calculate the mouth center in the reference image
        rx, ry = self._get_mouth_center(
            reference_landmarks, reference_image.shape[1], reference_image.shape[0]
        )

        # -- Calculate the mouth center in the image to be cropped
        cx, cy = self._get_mouth_center(
            crop_landmarks, crop_image.shape[1], crop_image.shape[0]
        )

        # -- Align the mouth of the image to be cropped with the reference
        cropped_image = self._align_mouth(crop_image, (cx, cy), (rx, ry))
        move_image_1 = self._align_mouth(move_image_1, (cx, cy), (rx, ry))
        move_image_2 = self._align_mouth(move_image_2, (cx, cy), (rx, ry))

        # --- 2nd face_mesh on the aligned image to detect the mouth in the new position
        aligned_landmarks = self._get_face_landmarks(face_mesh, cropped_image)

        # If detection fails, return the aligned image and an empty mask
        if aligned_landmarks is None:
            aligned_image_torch = self._numpy_to_torch(cropped_image)
            empty_mask = torch.zeros(
                (1, cropped_image.shape[0], cropped_image.shape[1])
            )
            return (aligned_image_torch.unsqueeze(0), empty_mask.unsqueeze(0))

        # Otherwise, create the mouth mask
        mouth_mask = self._create_mouth_mask(cropped_image, aligned_landmarks)

        cropped_image_torch = self._numpy_to_torch(cropped_image)
        mask_torch = self._numpy_to_torch(mouth_mask)
        move_image_1_torch = self._numpy_to_torch(move_image_1)
        move_image_2_torch = self._numpy_to_torch(move_image_2)

        # Return the cropped image
        return (
            cropped_image_torch.unsqueeze(0),
            mask_torch.unsqueeze(0),
            move_image_1_torch.unsqueeze(0),
            move_image_2_torch.unsqueeze(0),
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

    def _get_mouth_center(self, landmarks, width, height):
        """
        Calculate the position (x, y) of the mouth center in pixels.
        Example: the average of the outer corners of the mouth (indices 61 and 291).
        """
        left_x = landmarks[61].x * width
        left_y = landmarks[61].y * height

        right_x = landmarks[291].x * width
        right_y = landmarks[291].y * height

        center_x = (left_x + right_x) / 2.0
        center_y = (left_y + right_y) / 2.0

        return int(center_x), int(center_y)

    def _align_mouth(self, image_np, crop_mouth_center, ref_mouth_center):
        """
        Align the mouth of the 'image_np' based on the
        mouth coordinates of the reference image.
        - crop_mouth_center: (cx, cy) of the image to align
        - ref_mouth_center: (rx, ry) of the reference image
        """
        cx, cy = crop_mouth_center
        rx, ry = ref_mouth_center

        # Calculate the offset
        dx = rx - cx
        dy = ry - cy

        # Transformation matrix (translation)
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply translation
        height, width = image_np.shape[:2]
        aligned_image = cv2.warpAffine(
            image_np, M, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
        return aligned_image

    def _create_mouth_mask(self, aligned_image, aligned_landmarks):
        """
        Create a mask (H, W) of the mouth based on the landmarks
        detected in the aligned image. The outer lip indices are used.
        """
        height, width = aligned_image.shape[:2]

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
            x = int(aligned_landmarks[idx].x * width)
            y = int(aligned_landmarks[idx].y * height)
            points.append([x, y])

        # Convert to NumPy array for fillPoly
        points = np.array([points], dtype=np.int32)

        # Create a black mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the mouth in white
        cv2.fillPoly(mask, points, 255)

        return mask
