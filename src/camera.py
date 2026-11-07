import cv2

from .config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT
)


class Camera:

    def __init__(self, index=CAMERA_INDEX):

        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. "
                "Check camera permissions or try another camera index."
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

    def read(self):

        success, frame = self.cap.read()

        if not success or frame is None:
            return None

        # Mirror the camera for natural interaction.
        frame = cv2.flip(frame, 1)

        return frame

    def release(self):

        if self.cap is not None:
            self.cap.release()
