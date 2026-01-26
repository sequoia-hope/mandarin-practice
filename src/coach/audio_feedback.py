"""Audio feedback tones for gamification."""

import numpy as np
import sounddevice as sd


def generate_tone(frequency: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate a simple sine wave tone."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    # Apply fade in/out to avoid clicks
    fade_samples = int(0.01 * sample_rate)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return tone.astype(np.float32) * 0.5


def generate_success_sound() -> np.ndarray:
    """
    Generate a pleasant success chime.
    Two ascending tones - feels rewarding!
    """
    sr = 16000
    # C5 -> E5 (major third, happy sound)
    tone1 = generate_tone(523.25, 0.15, sr)  # C5
    tone2 = generate_tone(659.25, 0.2, sr)   # E5
    silence = np.zeros(int(0.05 * sr), dtype=np.float32)
    return np.concatenate([tone1, silence, tone2])


def generate_perfect_sound() -> np.ndarray:
    """
    Generate an extra rewarding sound for perfect scores.
    C-E-G arpeggio (major chord).
    """
    sr = 16000
    tone1 = generate_tone(523.25, 0.1, sr)   # C5
    tone2 = generate_tone(659.25, 0.1, sr)   # E5
    tone3 = generate_tone(783.99, 0.2, sr)   # G5
    silence = np.zeros(int(0.03 * sr), dtype=np.float32)
    return np.concatenate([tone1, silence, tone2, silence, tone3])


def generate_try_again_sound() -> np.ndarray:
    """
    Generate a gentle 'try again' tone.
    Descending minor second - not harsh, just indicates retry.
    """
    sr = 16000
    tone1 = generate_tone(440, 0.15, sr)     # A4
    tone2 = generate_tone(415.30, 0.15, sr)  # Ab4
    silence = np.zeros(int(0.03 * sr), dtype=np.float32)
    return np.concatenate([tone1, silence, tone2])


def generate_menu_sound() -> np.ndarray:
    """Generate a short blip for menu transitions."""
    sr = 16000
    return generate_tone(880, 0.08, sr)  # A5, short


def generate_listening_sound() -> np.ndarray:
    """Generate a soft beep to indicate 'listening'."""
    sr = 16000
    return generate_tone(660, 0.1, sr) * 0.7  # E5, quieter


def generate_error_sound() -> np.ndarray:
    """Generate an error/invalid input sound."""
    sr = 16000
    tone = generate_tone(220, 0.3, sr)  # A3, low
    return tone


class AudioFeedback:
    """Play audio feedback sounds."""

    def __init__(self, enabled: bool = True, volume: float = 0.7):
        self.enabled = enabled
        self.volume = volume

        # Pre-generate sounds
        self._sounds = {
            "success": generate_success_sound(),
            "perfect": generate_perfect_sound(),
            "try_again": generate_try_again_sound(),
            "menu": generate_menu_sound(),
            "listening": generate_listening_sound(),
            "error": generate_error_sound(),
        }

    def play(self, sound_name: str, blocking: bool = False):
        """Play a sound by name."""
        if not self.enabled:
            return

        if sound_name not in self._sounds:
            print(f"Unknown sound: {sound_name}")
            return

        audio = self._sounds[sound_name] * self.volume

        try:
            sd.play(audio, samplerate=16000)
            if blocking:
                sd.wait()
        except Exception as e:
            print(f"Audio playback error: {e}")

    def success(self):
        """Play success sound."""
        self.play("success")

    def perfect(self):
        """Play perfect score sound."""
        self.play("perfect")

    def try_again(self):
        """Play try again sound."""
        self.play("try_again")

    def menu(self):
        """Play menu transition sound."""
        self.play("menu")

    def listening(self):
        """Play listening indicator."""
        self.play("listening")

    def error(self):
        """Play error sound."""
        self.play("error")


# Quick test
if __name__ == "__main__":
    import time

    feedback = AudioFeedback()

    print("Testing audio feedback sounds...")

    print("  Success sound:")
    feedback.success()
    time.sleep(0.6)

    print("  Perfect sound:")
    feedback.perfect()
    time.sleep(0.8)

    print("  Try again sound:")
    feedback.try_again()
    time.sleep(0.5)

    print("  Menu sound:")
    feedback.menu()
    time.sleep(0.3)

    print("  Listening sound:")
    feedback.listening()
    time.sleep(0.3)

    print("  Error sound:")
    feedback.error()
    time.sleep(0.5)

    print("Done!")
