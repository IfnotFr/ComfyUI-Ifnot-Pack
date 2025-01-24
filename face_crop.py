import torch
import mediapipe as mp
import numpy as np
import cv2
import math


class FaceCrop:
    CATEGORY = "Ifnot Pack"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT",),
                "height": ("INT",),
                "zoom_factor": ("FLOAT",),
                "top_offset": ("FLOAT",),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "face_crop"

    def face_crop(
        self, image, width, height, zoom_factor, top_offset, extra_pnginfo=None
    ):
        """
        Main entry point of the node.
        """
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        output_cropped_images = []
        for image_torch in image:
            image = self._torch_to_numpy(image_torch)
            landmarks = self._get_face_landmarks(face_mesh, image)

            if landmarks is not None:
                # Get the face rotation angle
                angle = self._compute_rotation_angle(
                    landmarks, image.shape[1], image.shape[0]
                )

                # Get the center of the face bounding box
                cx, cy = self._compute_face_center(
                    landmarks, image.shape[1], image.shape[0]
                )

                # Rotate image
                image = self._rotate_image(image, angle, cx, cy)

                # Get new landmarks after rotation
                landmarks = self._get_face_landmarks(face_mesh, image)

                # Get the bouding box of the face
                x_min, y_min, x_max, y_max = self._get_bounding_box(
                    landmarks, image.shape[1], image.shape[0]
                )

                # Crop/Resize the face
                final_resized = self._auto_crop_face_and_resize(
                    image,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    width,
                    height,
                    zoom_factor,
                    top_offset,
                )

                # Convert to torch
                cropped_image_torch = self._convert_to_torch(final_resized)

                # Store extra PNG info (if needed)
                # extra_pnginfo["angle"] = angle

                # Close the face mesh
                face_mesh.close()

                # Return the cropped image
                return (cropped_image_torch.unsqueeze(0),)
            else:
                # Close the face mesh
                face_mesh.close()

                # Return the original image
                return (image_torch.unsqueeze(0),)

    def _torch_to_numpy(self, image_torch):
        """Convert a Torch image [C, H, W] to a NumPy RGB image [H, W, C]."""
        return (image_torch.cpu().numpy() * 255).astype("uint8")

    def _convert_to_torch(self, image):
        """Convert to a normalized torch tensor [C, H, W]."""
        return torch.from_numpy(image.astype("float32") / 255.0)

    def _get_face_landmarks(self, face_mesh, image):
        """Return the landmarks of the first face detected in the image."""
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
        return None

    def _compute_rotation_angle(self, landmarks, width, height):
        """
        Compute the angle of rotation to make the face horizontal.
        """
        left_eye_idx = 33
        right_eye_idx = 263

        left_eye_lm = landmarks[left_eye_idx]
        right_eye_lm = landmarks[right_eye_idx]

        left_eye_x, left_eye_y = left_eye_lm.x * width, left_eye_lm.y * height
        right_eye_x, right_eye_y = right_eye_lm.x * width, right_eye_lm.y * height

        dx = right_eye_x - left_eye_x
        dy = right_eye_y - left_eye_y

        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def _compute_face_center(self, landmarks, width, height):
        """Center of the face bounding box."""
        x_coords = [lm.x * width for lm in landmarks]
        y_coords = [lm.y * height for lm in landmarks]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        return cx, cy

    def _rotate_image(self, image, angle, cx, cy):
        """
        Rotate the image around the center (cx, cy) by the given angle.
        """
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        height, width = image.shape[:2]
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    def _auto_crop_face_and_resize(
        self,
        image,
        x_min,
        y_min,
        x_max,
        y_max,
        width=512,
        height=768,
        zoom_factor=2.0,
        top_offset=150,
    ):
        # Check if the face bounding box is valid
        face_w = x_max - x_min
        face_h = y_max - y_min
        if face_w <= 0 or face_h <= 0:
            # If not, return the original image
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        H, W = image.shape[:2]

        # Center the face bounding box
        face_cx = (x_min + x_max) / 2.0
        face_cy = (y_min + y_max) / 2.0

        # Horizontal crop
        final_width = zoom_factor * face_w
        half_fw = final_width / 2.0

        # Right and left coordinates
        left_f = face_cx - half_fw
        right_f = face_cx + half_fw

        # Clamp the coordinates
        left_f = int(math.floor(left_f))
        right_f = int(math.ceil(right_f))

        # Create black image with the same height
        crop_width = right_f - left_f
        if crop_width <= 0:
            # Security check if the crop width is negative
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        new_img = np.zeros((H, crop_width, 3), dtype=image.dtype)

        # Compute the overlap between the original image and the new image
        src_x1 = max(0, left_f)
        src_x2 = min(W, right_f)
        overlap_width = src_x2 - src_x1

        if overlap_width > 0:
            dst_x1 = src_x1 - left_f
            dst_x2 = dst_x1 + overlap_width

            # Copy the overlap region
            new_img[0:H, dst_x1:dst_x2] = image[0:H, src_x1:src_x2]

        # Resize the image to the final width
        final_w = width
        scale_factor = final_w / float(crop_width)
        final_h = int(round(H * scale_factor))
        resized_img = cv2.resize(
            new_img, (final_w, final_h), interpolation=cv2.INTER_AREA
        )

        # Compute the desired y_min
        scaled_y_min = y_min * scale_factor

        # Create the final output image
        target_h = height
        final_output = np.zeros((target_h, final_w, 3), dtype=resized_img.dtype)

        # Shift the image to the desired y_min
        desired_y_min = top_offset
        shift = desired_y_min - scaled_y_min  # si > 0 => on décale l'image vers le bas
        shift_int = int(round(shift))

        # Copy the image to the final output
        src_y1 = 0
        src_y2 = final_h
        dst_y1 = shift_int
        dst_y2 = dst_y1 + final_h

        # If dst_y1 < 0, we cut the top of resized_img
        if dst_y1 < 0:
            src_y1 = -dst_y1
            dst_y1 = 0

        # If dst_y2 > target_h, we cut the bottom of resized_img
        if dst_y2 > target_h:
            src_y2 -= dst_y2 - target_h
            dst_y2 = target_h

        # Copy the image
        if src_y1 < src_y2:
            final_output[dst_y1:dst_y2, 0:final_w] = resized_img[
                src_y1:src_y2, 0:final_w
            ]

        return final_output

    def _get_bounding_box(self, landmarks, width, height):
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
