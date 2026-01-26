"""Energy-based syllable alignment for segmenting audio into syllables."""

from typing import Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


class EnergyAligner:
    """
    Energy-based syllable aligner.

    Uses short-time energy analysis to find syllable boundaries.
    Works by detecting energy valleys (silence/low energy between syllables).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        min_syllable_ms: float = 80.0,
        max_syllable_ms: float = 800.0,
    ):
        """
        Initialize energy aligner.

        Args:
            sample_rate: Audio sample rate
            frame_length_ms: Frame length for energy computation in ms
            hop_length_ms: Hop length between frames in ms
            min_syllable_ms: Minimum syllable duration in ms
            max_syllable_ms: Maximum syllable duration in ms
        """
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length_ms * sample_rate / 1000)
        self.hop_length = int(hop_length_ms * sample_rate / 1000)
        self.min_syllable_samples = int(min_syllable_ms * sample_rate / 1000)
        self.max_syllable_samples = int(max_syllable_ms * sample_rate / 1000)

    def compute_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute short-time energy of audio.

        Args:
            audio: Audio samples (1D array)

        Returns:
            Energy contour (1D array, one value per frame)
        """
        # Ensure float32
        audio = audio.astype(np.float32)

        # Compute RMS energy per frame
        n_frames = (len(audio) - self.frame_length) // self.hop_length + 1
        energy = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = audio[start:end]
            energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)

        return energy

    def find_valleys(
        self,
        energy: np.ndarray,
        n_syllables: int,
        smooth_window: int = 5,
    ) -> list[int]:
        """
        Find energy valleys that serve as syllable boundaries.

        Args:
            energy: Energy contour
            n_syllables: Expected number of syllables
            smooth_window: Window size for smoothing energy

        Returns:
            List of valley frame indices
        """
        if n_syllables <= 0:
            return []

        # Smooth energy
        smoothed = uniform_filter1d(energy, size=smooth_window)

        # Invert energy to find valleys as peaks
        inverted = -smoothed

        # Find all potential valleys (peaks in inverted energy)
        # Use adaptive prominence based on energy range
        energy_range = smoothed.max() - smoothed.min()
        min_prominence = energy_range * 0.1 if energy_range > 0 else 0.01

        peaks, properties = find_peaks(
            inverted,
            distance=self.min_syllable_samples // self.hop_length,
            prominence=min_prominence,
        )

        if len(peaks) == 0:
            # No clear valleys found, use uniform segmentation
            return []

        # We need n_syllables - 1 boundaries
        n_boundaries = n_syllables - 1

        if len(peaks) >= n_boundaries:
            # Select the deepest valleys (highest prominence)
            prominences = properties["prominences"]
            sorted_indices = np.argsort(-prominences)[:n_boundaries]
            selected_peaks = sorted(peaks[sorted_indices])
            return list(selected_peaks)
        else:
            # Not enough valleys found, return what we have
            return list(peaks)

    def align(
        self,
        audio: np.ndarray,
        pinyin: list[str],
    ) -> list[tuple[float, float]]:
        """
        Align audio to expected syllables.

        Args:
            audio: Audio samples (1D array)
            pinyin: List of pinyin syllables (used only for count)

        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        n_syllables = len(pinyin)

        if n_syllables == 0:
            return []

        if n_syllables == 1:
            # Single syllable - use entire audio
            return [(0.0, len(audio) / self.sample_rate)]

        # Compute energy
        energy = self.compute_energy(audio)

        # Find valleys
        valleys = self.find_valleys(energy, n_syllables)

        # Convert valley frame indices to sample indices
        valley_samples = [v * self.hop_length for v in valleys]

        # Build boundaries
        boundaries = [0] + valley_samples + [len(audio)]

        # Ensure we have exactly n_syllables segments
        if len(boundaries) - 1 != n_syllables:
            # Fall back to uniform segmentation
            boundaries = self._uniform_boundaries(len(audio), n_syllables)

        # Convert to (start, end) tuples in seconds
        segments = []
        for i in range(len(boundaries) - 1):
            start_sec = boundaries[i] / self.sample_rate
            end_sec = boundaries[i + 1] / self.sample_rate
            segments.append((start_sec, end_sec))

        return segments

    def _uniform_boundaries(
        self,
        n_samples: int,
        n_syllables: int,
    ) -> list[int]:
        """Generate uniform segment boundaries as fallback."""
        return [int(i * n_samples / n_syllables) for i in range(n_syllables + 1)]

    def align_with_durations(
        self,
        audio: np.ndarray,
        pinyin: list[str],
        expected_durations: Optional[list[float]] = None,
    ) -> list[tuple[float, float]]:
        """
        Align audio considering expected syllable durations.

        Args:
            audio: Audio samples
            pinyin: List of pinyin syllables
            expected_durations: Optional list of relative durations (weights)

        Returns:
            List of (start_time, end_time) tuples
        """
        if expected_durations is None:
            return self.align(audio, pinyin)

        n_syllables = len(pinyin)
        total_duration = len(audio) / self.sample_rate

        # Normalize durations
        total_weight = sum(expected_durations)
        normalized = [d / total_weight for d in expected_durations]

        # Calculate segment times
        segments = []
        current_time = 0.0

        for i, weight in enumerate(normalized):
            duration = weight * total_duration
            segments.append((current_time, current_time + duration))
            current_time += duration

        return segments


class VowelAligner(EnergyAligner):
    """
    Enhanced aligner that also considers vowel detection.

    Uses spectral features to identify vowel nuclei for more accurate segmentation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._spectral_enabled = True

    def compute_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral centroid as vowel indicator.

        Vowels tend to have lower spectral centroids than consonants.
        """
        try:
            import librosa
        except ImportError:
            self._spectral_enabled = False
            return np.zeros(1)

        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )[0]

        return centroid

    def find_vowel_nuclei(
        self,
        audio: np.ndarray,
        n_syllables: int,
    ) -> list[int]:
        """
        Find vowel nuclei (syllable centers) using spectral analysis.

        Args:
            audio: Audio samples
            n_syllables: Expected number of syllables

        Returns:
            List of frame indices corresponding to vowel nuclei
        """
        if not self._spectral_enabled:
            return []

        # Compute energy and spectral centroid
        energy = self.compute_energy(audio)
        centroid = self.compute_spectral_centroid(audio)

        # Align lengths
        min_len = min(len(energy), len(centroid))
        energy = energy[:min_len]
        centroid = centroid[:min_len]

        # Vowel score: high energy, low spectral centroid
        # Normalize both
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)
        centroid_norm = (centroid - centroid.min()) / (centroid.max() - centroid.min() + 1e-10)

        vowel_score = energy_norm * (1 - centroid_norm * 0.5)

        # Find peaks (vowel nuclei)
        peaks, _ = find_peaks(
            vowel_score,
            distance=self.min_syllable_samples // self.hop_length,
            height=0.3,
        )

        # Select top n_syllables peaks
        if len(peaks) >= n_syllables:
            scores = vowel_score[peaks]
            top_indices = np.argsort(-scores)[:n_syllables]
            return sorted(peaks[top_indices])

        return list(peaks)


if __name__ == "__main__":
    # Test alignment
    print("Testing energy-based alignment...")

    # Generate test audio with two syllables
    sr = 16000
    duration = 1.0

    # Simulate "ni hao" - two syllables with gap in between
    t = np.linspace(0, duration, int(sr * duration))

    # First syllable (0-0.4s)
    env1 = np.exp(-((t - 0.2) ** 2) / 0.02)
    # Second syllable (0.5-0.9s)
    env2 = np.exp(-((t - 0.7) ** 2) / 0.02)

    audio = (env1 + env2) * np.sin(2 * np.pi * 200 * t)
    audio = audio.astype(np.float32)

    aligner = EnergyAligner()
    segments = aligner.align(audio, ["ni", "hao"])

    print(f"Detected segments for 2 syllables:")
    for i, (start, end) in enumerate(segments):
        print(f"  Syllable {i+1}: {start:.3f}s - {end:.3f}s ({(end-start)*1000:.0f}ms)")

    # Test with 4 syllables
    segments = aligner.align(audio, ["ni", "hao", "ma", "a"])
    print(f"\nForced 4 syllable alignment (uniform fallback):")
    for i, (start, end) in enumerate(segments):
        print(f"  Syllable {i+1}: {start:.3f}s - {end:.3f}s")
