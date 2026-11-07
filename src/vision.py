import cv2
import mediapipe as mp

from .config import (
    HAND_MODEL,
    POSE_MODEL,
    NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_HAND_TRACKING_CONFIDENCE,
    MIN_POSE_DETECTION_CONFIDENCE,
    MIN_POSE_PRESENCE_CONFIDENCE,
    MIN_POSE_TRACKING_CONFIDENCE,
)


# MediaPipe Hand Landmarker indices:
# wrist=0, thumb tip=4, index tip=8, middle tip=12, ring tip=16, pinky tip=20.
FINGERTIP_INDICES = (4, 8, 12, 16, 20)

# These palm landmarks give a steadier hand-position point than the wrist.
# The main control point is the average of the four MCP knuckles.
PALM_CENTER_INDICES = (5, 9, 13, 17)


def get_palm_center(hand_landmarks):
    """Return normalized (x, y) palm-center coordinates."""
    x = sum(hand_landmarks[i].x for i in PALM_CENTER_INDICES) / len(
        PALM_CENTER_INDICES
    )
    y = sum(hand_landmarks[i].y for i in PALM_CENTER_INDICES) / len(
        PALM_CENTER_INDICES
    )
    return float(x), float(y)


def get_hand_source(hand_result, hand_index):
    """Return a stable left/right source ID from MediaPipe handedness."""
    handedness = getattr(hand_result, "handedness", ())
    if hand_index < len(handedness) and handedness[hand_index]:
        category = handedness[hand_index][0]
        label = getattr(category, "category_name", "") or getattr(
            category, "display_name", ""
        )
        label = str(label).lower()
        if label in ("left", "right"):
            return f"{label}_hand"

    # Keep processing usable with result objects that omit handedness.
    return f"hand_{hand_index}"


class VisionEngine:
    def __init__(self):
        if not HAND_MODEL.is_file():
            raise FileNotFoundError(
                f"Hand model not found: {HAND_MODEL}\n"
                "Place hand_landmarker.task in the models folder."
            )
        if not POSE_MODEL.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {POSE_MODEL}\n"
                "Place pose_landmarker_full.task in the models folder."
            )

        BaseOptions = mp.tasks.BaseOptions
        RunningMode = mp.tasks.vision.RunningMode
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=RunningMode.IMAGE,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_HAND_TRACKING_CONFIDENCE,
        )
        self.hand_detector = HandLandmarker.create_from_options(hand_options)

        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_POSE_TRACKING_CONFIDENCE,
        )
        self.pose_detector = PoseLandmarker.create_from_options(pose_options)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = self.hand_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)
        return hand_result, pose_result

    @staticmethod
    def draw(frame, hand_result, pose_result):
        height, width = frame.shape[:2]

        for hand_index, hand in enumerate(hand_result.hand_landmarks):
            # Draw all hand landmarks.
            for landmark in hand:
                px = int(landmark.x * width)
                py = int(landmark.y * height)
                cv2.circle(frame, (px, py), 2, (0, 220, 0), -1)

            # Highlight fingertips.
            for index in FINGERTIP_INDICES:
                tip = hand[index]
                tx, ty = int(tip.x * width), int(tip.y * height)
                cv2.circle(frame, (tx, ty), 6, (0, 255, 255), 2)

            # Highlight the palm-center point used for zone mapping.
            cx, cy = get_palm_center(hand)
            center_px, center_py = int(cx * width), int(cy * height)
            cv2.circle(frame, (center_px, center_py), 9, (255, 255, 0), 2)
            cv2.putText(
                frame,
                get_hand_source(hand_result, hand_index).replace(
                    "_", " ").title(),
                (center_px + 8, center_py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw selected body landmarks for visual feedback.
        if pose_result.pose_landmarks:
            pose = pose_result.pose_landmarks[0]
            selected = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
            for index in selected:
                landmark = pose[index]
                px, py = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (px, py), 3, (0, 140, 255), -1)

            connections = (
                (11, 13), (13, 15), (12, 14), (14, 16), (11, 12),
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                (24, 26), (26, 28),
            )
            for start, end in connections:
                p1, p2 = pose[start], pose[end]
                cv2.line(
                    frame,
                    (int(p1.x * width), int(p1.y * height)),
                    (int(p2.x * width), int(p2.y * height)),
                    (0, 140, 255),
                    2,
                )

        return frame

    def close(self):
        self.hand_detector.close()
        self.pose_detector.close()
