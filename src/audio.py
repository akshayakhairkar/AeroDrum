import math
import wave

import numpy as np
import pygame

from .config import (
    SAMPLE_DIR,
    AUDIO_FREQUENCY,
    AUDIO_SIZE,
    AUDIO_CHANNELS,
    AUDIO_BUFFER,
    MASTER_VOLUME,
)

ZONE_FILES = {
    "kick": "kick.wav",
    "snare": "snare.wav",
    "hi_hat": "hihat.wav",
    "high_tom": "high_tom.wav",
    "mid_tom": "mid_tom.wav",
    "floor_tom": "floor_tom.wav",
    "crash": "crash.wav",
    "ride": "ride.wav",
}

FALLBACK_FILES = {
    "high_tom": "tom.wav",
    "mid_tom": "tom.wav",
    "floor_tom": "tom.wav",
}


def generate_sample(path, frequency, duration, decay, noise=False, seed=4):
    """Generate a simple demo WAV only when a required file is missing."""
    sample_rate = AUDIO_FREQUENCY
    count = int(sample_rate * duration)
    times = np.arange(count, dtype=np.float32) / sample_rate

    if noise:
        rng = np.random.default_rng(seed)
        signal = rng.normal(0.0, 1.0, count).astype(np.float32)
    else:
        signal = np.sin(2 * math.pi * frequency * times).astype(np.float32)

    signal *= np.exp(-decay * times).astype(np.float32)
    signal = np.clip(signal * 0.55, -1.0, 1.0)
    stereo = np.repeat(signal[:, None], 2, axis=1)
    pcm = (stereo * 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def ensure_demo_samples():
    """Ensure baseline samples exist; do not overwrite existing user samples."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    samples = {
        "kick.wav": (70, 0.45, 5.5, False, 1),
        "snare.wav": (180, 0.22, 12.0, True, 2),
        "hihat.wav": (4500, 0.10, 28.0, True, 3),
        "tom.wav": (150, 0.35, 7.0, False, 4),
        "crash.wav": (2500, 0.65, 7.0, True, 5),
        "ride.wav": (3200, 0.55, 8.0, True, 6),
        "high_tom.wav": (220, 0.24, 9.0, False, 7),
        "mid_tom.wav": (165, 0.30, 7.5, False, 8),
        "floor_tom.wav": (100, 0.42, 5.5, False, 9),
    }

    for filename, values in samples.items():
        path = SAMPLE_DIR / filename
        if not path.exists():
            generate_sample(path, *values)


class DrumSampler:
    def __init__(self):
        ensure_demo_samples()

        pygame.mixer.pre_init(
            frequency=AUDIO_FREQUENCY,
            size=AUDIO_SIZE,
            channels=AUDIO_CHANNELS,
            buffer=AUDIO_BUFFER,
        )
        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)

        self.sounds = {}
        self.loaded_files = {}

        for zone, filename in ZONE_FILES.items():
            path = SAMPLE_DIR / filename
            if not path.exists() and zone in FALLBACK_FILES:
                path = SAMPLE_DIR / FALLBACK_FILES[zone]

            try:
                self.sounds[zone] = pygame.mixer.Sound(str(path))
                self.loaded_files[zone] = path.name
                print(f"Loaded {zone}: {path.name}")
            except (pygame.error, OSError) as error:
                print(f"Could not load sample for {zone} ({path}): {error}")

    def play(self, event):
        sound = self.sounds.get(event.zone)
        if sound is None:
            print(f"No sound mapped for zone: {event.zone}")
            return

        # Scale loudness from detected strike velocity.
        volume = max(0.15, min(1.0, float(event.velocity) * 3.0))
        sound.set_volume(MASTER_VOLUME * volume)
        sound.play()

    def close(self):
        if pygame.mixer.get_init():
            pygame.mixer.quit()
