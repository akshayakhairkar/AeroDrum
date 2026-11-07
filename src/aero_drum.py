import cv2
import time

from .camera import Camera
from .vision import VisionEngine, get_hand_source, get_palm_center
from .zones import zone_for_point, draw_zones
from .strike_detector import StrikeDetector
from .audio import DrumSampler
from .event_queue import GestureEventQueue
from .config import SHOW_LANDMARKS, SHOW_ZONES, SHOW_DEBUG


def main():
    camera = None
    vision = None
    sampler = None

    try:
        camera = Camera()
        vision = VisionEngine()
        detector = StrikeDetector()
        sampler = DrumSampler()
        event_queue = GestureEventQueue(max_size=256)

        print("AeroDrum started successfully.")
        print("Interaction point: palm center (not wrist).")
        print("Eight virtual drum zones are enabled.")
        print("Press Q or ESC to exit.")

        previous_frame_time = time.perf_counter()
        fps = 0.0
        last_event = None
        visible_sources = set()
        previous_points = {}

        while True:
            frame = camera.read()
            if frame is None:
                print("Warning: Could not read frame from camera.")
                continue

            hand_result, pose_result = vision.process(frame)

            if SHOW_LANDMARKS:
                frame = vision.draw(frame, hand_result, pose_result)

            # Track the palm-center point for each detected hand.
            # This uses the hand's position, not landmark 0 (the wrist).
            current_sources = set()

            if hand_result.hand_landmarks:
                for hand_index, hand in enumerate(hand_result.hand_landmarks):
                    source = get_hand_source(hand_result, hand_index)
                    current_sources.add(source)

                    strike_x, strike_y = get_palm_center(hand)
                    previous_point = previous_points.get(source)
                    trajectory = (0.0, 0.0)
                    if previous_point is not None:
                        trajectory = (
                            strike_x - previous_point[0],
                            strike_y - previous_point[1],
                        )
                    previous_points[source] = (strike_x, strike_y)
                    drum_name = zone_for_point(
                        strike_x,
                        strike_y,
                        trajectory=trajectory,
                    )

                    # MediaPipe Tasks HandLandmarker does not provide a single
                    # per-hand confidence in the returned landmark list.
                    # Use 1.0 here; detector presence thresholds are configured
                    # in VisionEngine.
                    event = detector.update(
                        source=source,
                        x=strike_x,
                        y=strike_y,
                        zone=drum_name,
                        confidence=1.0,
                    )

                    if SHOW_DEBUG:
                        h, w = frame.shape[:2]
                        px, py = int(strike_x * w), int(strike_y * h)
                        zone_text = (
                            drum_name.upper().replace("_", " ")
                            if drum_name
                            else "NO ZONE"
                        )
                        cv2.putText(
                            frame,
                            zone_text,
                            (px + 8, py + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    if event is None:
                        continue

                    last_event = event
                    event_queue.push(event)

            # Reset motion history for a hand that disappeared, preventing
            # a large false velocity when it reappears.
            for missing_source in visible_sources - current_sources:
                detector.reset_source(missing_source)
                previous_points.pop(missing_source, None)
            visible_sources = current_sources

            while not event_queue.empty():
                queued_event = event_queue.pop()
                if queued_event is not None:
                    sampler.play(queued_event)

            if SHOW_ZONES:
                highlighted_zones = ()
                if last_event is not None:
                    if time.perf_counter() - last_event.timestamp < 0.12:
                        highlighted_zones = (last_event.zone,)
                frame = draw_zones(frame, highlighted_zones)

            current_frame_time = time.perf_counter()
            delta_time = current_frame_time - previous_frame_time
            if delta_time > 0:
                fps = (fps * 0.9) + ((1.0 / delta_time) * 0.1)
            previous_frame_time = current_frame_time

            if SHOW_DEBUG:
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, f"Queue: {event_queue.size()}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,
                                                     255, 255), 2, cv2.LINE_AA
                )

                if last_event is not None:
                    event_text = (
                        f"Last: {last_event.zone.upper()} "
                        f"| V: {last_event.velocity:.2f}"
                    )
                    cv2.putText(
                        frame, event_text, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,
                                                        255, 255), 2, cv2.LINE_AA
                    )

                stats = event_queue.statistics()
                cv2.putText(
                    frame,
                    f"Events: {stats['pushed']} | Dropped: {stats['dropped']}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA
                )

            cv2.imshow("AeroDrum", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\nAeroDrum stopped by user.")
    except Exception as error:
        print("\nAeroDrum encountered an error:")
        print(error)
        raise
    finally:
        if camera is not None:
            camera.release()
        if vision is not None:
            vision.close()
        if sampler is not None:
            sampler.close()
        cv2.destroyAllWindows()
        print("AeroDrum closed successfully.")


if __name__ == "__main__":
    main()
