from time import perf_counter

from .config import (
    POSITION_DEADBAND,
    POSITION_SMOOTHING_ALPHA,
    RETRIGGER_UPWARD_MOVEMENT,
    VELOCITY_THRESHOLD,
    MIN_DOWNWARD_MOVEMENT,
    COOLDOWN_SECONDS,
    MIN_CONFIDENCE,
)
from .events import GestureEvent


class StrikeDetector:
    """Detect downward strikes using the tracked palm-center point."""

    def __init__(
        self,
        velocity_threshold=VELOCITY_THRESHOLD,
        cooldown_seconds=COOLDOWN_SECONDS,
        rearm_upward_movement=RETRIGGER_UPWARD_MOVEMENT,
        min_downward_movement=MIN_DOWNWARD_MOVEMENT,
        min_confidence=MIN_CONFIDENCE,
    ):
        self.velocity_threshold = float(velocity_threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.rearm_upward_movement = float(rearm_upward_movement)
        self.min_downward_movement = float(min_downward_movement)
        self.min_confidence = float(min_confidence)
        self.previous_position = {}
        self.previous_time = {}
        self.last_hit = {}
        self.armed = {}
        self.current_zone = {}

    def reset_source(self, source):
        """Reset motion history when a hand temporarily disappears."""
        self.previous_position.pop(source, None)
        self.previous_time.pop(source, None)
        self.current_zone.pop(source, None)
        self.last_hit.pop(source, None)
        self.armed.pop(source, None)

    def update(self, source, x, y, zone, confidence=1.0):
        """Return a GestureEvent when a valid downward strike is detected."""
        now = perf_counter()
        current = (float(x), float(y))

        if source not in self.previous_position:
            self.previous_position[source] = current
            self.previous_time[source] = now
            self.current_zone[source] = zone
            return None

        previous = self.previous_position[source]
        previous_time = self.previous_time[source]

        # Smooth small landmark jitter while retaining quick intentional motion.
        alpha = POSITION_SMOOTHING_ALPHA
        filtered = (
            previous[0] + alpha * (current[0] - previous[0]),
            previous[1] + alpha * (current[1] - previous[1]),
        )
        self.previous_position[source] = filtered
        self.previous_time[source] = now

        self.current_zone[source] = zone
        if zone is None:
            self.armed[source] = True
            return None

        if confidence < self.min_confidence:
            return None

        dt = max(now - previous_time, 1e-4)
        dx = filtered[0] - previous[0]
        dy = filtered[1] - previous[1]
        downward_velocity = dy / dt

        self.armed.setdefault(source, True)
        if dy <= -self.rearm_upward_movement:
            self.armed[source] = True

        if abs(dx) < POSITION_DEADBAND and abs(dy) < POSITION_DEADBAND:
            return None

        # In image coordinates, positive dy means movement downward.
        last_hit_time = self.last_hit.get(source, -1e9)
        cooldown_passed = (now - last_hit_time) >= self.cooldown_seconds

        if (
            self.armed[source]
            and downward_velocity >= self.velocity_threshold
            and dy >= self.min_downward_movement
            and cooldown_passed
        ):
            self.last_hit[source] = now
            self.armed[source] = False
            return GestureEvent(
                timestamp=now,
                zone=zone,
                velocity=min(float(downward_velocity), 1.0),
                source=source,
                confidence=float(confidence),
            )

        return None
