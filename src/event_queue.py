from collections import deque
from threading import Lock


class GestureEventQueue:
    def __init__(self, max_size=256):
        self.queue = deque(maxlen=max_size)
        self.lock = Lock()

        self.total_pushed = 0
        self.total_popped = 0
        self.total_dropped = 0

    def push(self, event):
        with self.lock:
            if len(self.queue) >= self.queue.maxlen:
                self.total_dropped += 1

            self.queue.append(event)
            self.total_pushed += 1

    def pop(self):
        with self.lock:
            if not self.queue:
                return None

            event = self.queue.popleft()
            self.total_popped += 1

            return event

    def empty(self):
        """Return True when the queue contains no events."""
        with self.lock:
            return len(self.queue) == 0

    def size(self):
        """Return the current number of queued events."""
        with self.lock:
            return len(self.queue)

    def statistics(self):
        """Return queue statistics for debugging."""
        with self.lock:
            return {
                "queued": len(self.queue),
                "pushed": self.total_pushed,
                "popped": self.total_popped,
                "dropped": self.total_dropped
            }

    def clear(self):
        """Remove all pending events."""
        with self.lock:
            self.queue.clear()
