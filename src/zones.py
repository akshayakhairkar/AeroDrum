from dataclasses import dataclass
import cv2


GRID_X = (0.03, 0.35, 0.67, 0.98)
GRID_Y = (0.16, 0.48, 0.66, 0.98)


@dataclass(frozen=True)
class DrumZone:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    label: str


# Normalized coordinates (0..1), so the layout adapts to the camera resolution.
# Every occupied cell uses the same shared grid boundaries. The top-middle
# cell is intentionally omitted from ZONES.
ZONES = [
    # Top row: cymbals
    DrumZone("crash", GRID_X[0], GRID_Y[0], GRID_X[1], GRID_Y[1], "CRASH"),
    DrumZone("ride", GRID_X[2], GRID_Y[0], GRID_X[3], GRID_Y[1], "RIDE"),

    # Middle row: hi-hat and two rack toms
    DrumZone("hi_hat", GRID_X[0], GRID_Y[1], GRID_X[1], GRID_Y[2], "HI-HAT"),
    DrumZone("high_tom", GRID_X[1], GRID_Y[1], GRID_X[2], GRID_Y[2], "TOM 1"),
    DrumZone("mid_tom", GRID_X[2], GRID_Y[1], GRID_X[3], GRID_Y[2], "TOM 2"),

    # Bottom row: snare, bass, and floor tom
    DrumZone("snare", GRID_X[0], GRID_Y[2], GRID_X[1], GRID_Y[3], "SNARE"),
    DrumZone("kick", GRID_X[1], GRID_Y[2], GRID_X[2], GRID_Y[3], "BASS"),
    DrumZone("floor_tom", GRID_X[2], GRID_Y[2], GRID_X[3], GRID_Y[3], "FLOOR"),
]

ZONE_BY_CELL = {
    (0, 0): "crash",
    (2, 0): "ride",
    (0, 1): "hi_hat",
    (1, 1): "high_tom",
    (2, 1): "mid_tom",
    (0, 2): "snare",
    (1, 2): "kick",
    (2, 2): "floor_tom",
}


FORGIVING_ZONE_RADIUS = 0.30
TRAJECTORY_LOOKAHEAD = 1.5


def zone_for_point(x, y, trajectory=(0.0, 0.0)):
    """Return the nearest likely drum for a strike position and trajectory."""
    if GRID_X[1] <= x <= GRID_X[2] and GRID_Y[0] <= y <= GRID_Y[1]:
        return None

    trajectory_x, trajectory_y = trajectory
    estimate_x = x + trajectory_x * TRAJECTORY_LOOKAHEAD
    estimate_y = y + trajectory_y * TRAJECTORY_LOOKAHEAD

    nearest_zone = None
    nearest_distance = float("inf")

    for zone in ZONES:
        center_x = (zone.x1 + zone.x2) / 2
        center_y = (zone.y1 + zone.y2) / 2
        distance = (
            (estimate_x - center_x) ** 2 + (estimate_y - center_y) ** 2
        ) ** 0.5
        if distance < nearest_distance:
            nearest_zone = zone.name
            nearest_distance = distance

    if nearest_distance > FORGIVING_ZONE_RADIUS:
        return None
    return nearest_zone


def draw_zones(frame, highlighted_zones=()):
    """Draw all virtual drum zones on the camera frame."""
    height, width = frame.shape[:2]
    highlighted_zones = set(highlighted_zones)

    for row in range(3):
        for column in range(3):
            x1 = int(GRID_X[column] * width)
            y1 = int(GRID_Y[row] * height)
            x2 = int(GRID_X[column + 1] * width)
            y2 = int(GRID_Y[row + 1] * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)

    for zone in ZONES:
        x1 = int(zone.x1 * width)
        y1 = int(zone.y1 * height)
        x2 = int(zone.x2 * width)
        y2 = int(zone.y2 * height)

        if zone.name in highlighted_zones:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Keep the label inside the image and readable.
        label_y = max(18, y1 - 7)
        (text_w, text_h), _ = cv2.getTextSize(
            zone.label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_top = max(0, label_y - text_h - 7)
        label_right = min(width - 1, x1 + text_w + 8)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (label_right, label_y + 3),
            (30, 30, 30),
            -1,
        )
        cv2.putText(
            frame,
            zone.label,
            (x1 + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame
