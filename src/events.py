from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class GestureEvent:

    timestamp: float

    zone: str

    velocity: float

    source: str

    confidence: float


def create_event(
    zone,
    velocity,
    source,
    confidence
):

    return GestureEvent(

        timestamp=perf_counter(),

        zone=zone,

        velocity=float(velocity),

        source=source,

        confidence=float(confidence)
    )
