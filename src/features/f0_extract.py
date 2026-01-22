"""F0 (fundamental frequency) extraction using CREPE and pYIN."""

from typing import Optional, Literal

import numpy as np
import torch


class F0Extractor:
    """
    Extract F0 (pitch) features from audio.

    Uses pYIN for real-time (faster) and CREPE-tiny for training (more accurate).
    Returns F0 contour, delta-F0, and confidence values.
    """

    def __init__(
        self,
        method: Literal["pyin", "crepe"] = "pyin",
        sample_rate: int = 16000,
        hop_length: int = 160,  # 10ms at 16kHz
        fmin: float = 50.0,
        fmax: float = 500.0,
        crepe_model: str = "tiny",
        device: Optional[str] = None,
    ):
        """
        Initialize F0 extractor.

        Args:
            method: Extraction method ('pyin' for fast, 'crepe' for accurate)
            sample_rate: Audio sample rate
            hop_length: Hop length in samples
            fmin: Minimum F0 frequency
            fmax: Maximum F0 frequency
            crepe_model: CREPE model size ('tiny', 'small', 'medium', 'large', 'full')
            device: Device for CREPE (GPU accelerated)
        """
        self.method = method
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.crepe_model = crepe_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Output dimension: f0, delta_f0, confidence
        self.output_dim = 3

    def extract_pyin(
        self,
        audio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract F0 using pYIN algorithm.

        Fast method suitable for real-time use.
        """
        import librosa

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fill_na=0.0,
        )

        # voiced_prob serves as confidence
        confidence = voiced_prob

        return f0, voiced_flag, confidence

    def extract_crepe(
        self,
        audio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract F0 using CREPE neural network.

        More accurate but slower, suitable for training data.
        """
        import crepe

        # CREPE expects audio at its native sample rate
        time, f0, confidence, activation = crepe.predict(
            audio,
            self.sample_rate,
            model_capacity=self.crepe_model,
            step_size=self.hop_length * 1000 / self.sample_rate,  # in ms
            viterbi=True,
            device=self.device,
        )

        # Filter by confidence and frequency range
        voiced_flag = (confidence > 0.5) & (f0 >= self.fmin) & (f0 <= self.fmax)

        return f0, voiced_flag, confidence

    def extract(
        self,
        audio: np.ndarray,
        normalize: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Extract F0 features from audio.

        Args:
            audio: Audio samples (1D numpy array)
            normalize: Whether to normalize F0 to semitones

        Returns:
            Dict with:
                - 'f0': F0 contour
                - 'delta_f0': First derivative of F0
                - 'confidence': Voicing confidence
                - 'features': Stacked features (time, 3)
        """
        # Extract raw F0
        if self.method == "pyin":
            f0, voiced_flag, confidence = self.extract_pyin(audio)
        else:
            f0, voiced_flag, confidence = self.extract_crepe(audio)

        # Handle NaN values
        f0 = np.nan_to_num(f0, nan=0.0)
        confidence = np.nan_to_num(confidence, nan=0.0)

        # Normalize F0 to semitones relative to mean (more invariant to speaker)
        if normalize and np.any(f0 > 0):
            voiced_f0 = f0[f0 > 0]
            if len(voiced_f0) > 0:
                mean_f0 = np.mean(voiced_f0)
                # Convert to semitones: 12 * log2(f/ref)
                f0_normalized = np.where(
                    f0 > 0,
                    12 * np.log2(f0 / mean_f0 + 1e-8),
                    0.0,
                )
            else:
                f0_normalized = f0
        else:
            f0_normalized = f0

        # Compute delta F0 (first derivative)
        delta_f0 = np.gradient(f0_normalized)

        # Stack features
        features = np.stack([f0_normalized, delta_f0, confidence], axis=-1)

        return {
            "f0": f0,
            "f0_normalized": f0_normalized,
            "delta_f0": delta_f0,
            "confidence": confidence,
            "voiced_flag": voiced_flag.astype(np.float32),
            "features": features.astype(np.float32),
        }

    def extract_batch(
        self,
        audio_batch: list[np.ndarray],
        normalize: bool = True,
    ) -> list[dict[str, np.ndarray]]:
        """Extract F0 features from a batch of audio."""
        return [self.extract(audio, normalize) for audio in audio_batch]

    def to_tensor(
        self,
        features: dict[str, np.ndarray],
        device: Optional[str] = None,
    ) -> dict[str, torch.Tensor]:
        """Convert numpy features to PyTorch tensors."""
        if device is None:
            device = self.device

        return {
            key: torch.from_numpy(value).to(device) if isinstance(value, np.ndarray) else value
            for key, value in features.items()
        }

    def resample_to_frames(
        self,
        f0_features: np.ndarray,
        target_length: int,
    ) -> np.ndarray:
        """
        Resample F0 features to match MMS output length.

        Args:
            f0_features: F0 features (time, 3)
            target_length: Target number of frames

        Returns:
            Resampled features (target_length, 3)
        """
        from scipy import interpolate

        current_length = len(f0_features)

        if current_length == target_length:
            return f0_features

        # Create interpolation function for each feature dimension
        x_original = np.linspace(0, 1, current_length)
        x_target = np.linspace(0, 1, target_length)

        resampled = np.zeros((target_length, f0_features.shape[-1]))

        for i in range(f0_features.shape[-1]):
            f = interpolate.interp1d(
                x_original,
                f0_features[:, i],
                kind="linear",
                fill_value="extrapolate",
            )
            resampled[:, i] = f(x_target)

        return resampled.astype(np.float32)


class F0ExtractorRT(F0Extractor):
    """
    Real-time optimized F0 extractor.

    Uses pYIN with smaller windows for lower latency.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 80,  # 5ms at 16kHz for lower latency
        **kwargs,
    ):
        super().__init__(
            method="pyin",
            sample_rate=sample_rate,
            hop_length=hop_length,
            **kwargs,
        )


if __name__ == "__main__":
    # Test F0 extraction
    print("Testing F0 extraction...")

    # Generate test audio with a simple tone
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate a Mandarin tone 4 (falling): start at 400Hz, fall to 200Hz
    f0_contour = np.linspace(400, 200, len(t))
    audio = np.sin(2 * np.pi * np.cumsum(f0_contour) / sr).astype(np.float32)

    extractor = F0Extractor(method="pyin")
    features = extractor.extract(audio)

    print(f"F0 shape: {features['f0'].shape}")
    print(f"Delta F0 shape: {features['delta_f0'].shape}")
    print(f"Confidence shape: {features['confidence'].shape}")
    print(f"Features shape: {features['features'].shape}")
    print(f"F0 range: {features['f0'].min():.1f} - {features['f0'].max():.1f} Hz")
