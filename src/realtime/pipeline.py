"""End-to-end real-time inference pipeline for tone classification."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..features.mms_encoder import MMSEncoder
from ..features.f0_extract import F0Extractor
from ..model.classifier import ToneClassifier, TONE_NAMES
from .audio_capture import UtteranceDetector
from .vad import SileroVAD


@dataclass
class ToneResult:
    """Result of tone classification."""

    tone: int  # 0-4
    confidence: float
    tone_name: str
    f0_contour: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    audio_duration_ms: float = 0.0


class RealtimePipeline:
    """
    Real-time tone classification pipeline.

    Combines:
    - Audio capture with VAD
    - MMS feature extraction
    - F0 feature extraction
    - Tone classification

    Target latency: <300ms end-to-end
    """

    def __init__(
        self,
        classifier_checkpoint: Optional[str | Path] = None,
        mms_model: str = "facebook/mms-300m",
        use_f0: bool = True,
        use_light_encoder: bool = True,
        vad_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """
        Initialize pipeline.

        Args:
            classifier_checkpoint: Path to trained classifier checkpoint
            mms_model: MMS model identifier
            use_f0: Whether to use F0 features
            use_light_encoder: Use lighter MMS encoder for speed
            vad_threshold: VAD speech probability threshold
            device: Torch device
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_f0 = use_f0

        print(f"Initializing pipeline on {device}...")

        # Load MMS encoder
        print("Loading MMS encoder...")
        if use_light_encoder:
            from ..features.mms_encoder import MMSEncoderLight
            self.mms_encoder = MMSEncoderLight(mms_model, device=device)
        else:
            self.mms_encoder = MMSEncoder(mms_model, device=device)

        # Load F0 extractor
        if use_f0:
            print("Loading F0 extractor...")
            self.f0_extractor = F0Extractor(method="pyin")
        else:
            self.f0_extractor = None

        # Load classifier
        print("Loading classifier...")
        if classifier_checkpoint:
            from ..model.train import load_model
            self.classifier = load_model(classifier_checkpoint, device)
        else:
            # Create default classifier (untrained)
            self.classifier = ToneClassifier(
                input_dim=self.mms_encoder.output_dim,
                use_f0=use_f0,
            ).to(device)

        self.classifier.eval()

        # VAD for standalone processing
        self.vad = SileroVAD(threshold=vad_threshold)

        print("Pipeline ready.")

    @torch.no_grad()
    def classify(self, audio: np.ndarray) -> ToneResult:
        """
        Classify tone from audio segment.

        Args:
            audio: Audio samples (numpy array, mono, 16kHz)

        Returns:
            ToneResult with classification and timing info
        """
        start_time = time.perf_counter()
        audio_duration_ms = len(audio) / 16000 * 1000

        # Ensure float32
        audio = audio.astype(np.float32)

        # Extract MMS features
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        mms_features = self.mms_encoder(audio_tensor)

        # Handle different encoder outputs
        if isinstance(mms_features, dict):
            mms_features = mms_features.get("pooled", mms_features.get("combined"))
            if mms_features.dim() == 3:
                mms_features = mms_features.mean(dim=1)

        # Extract F0 features
        f0_features = None
        f0_contour = None
        if self.use_f0 and self.f0_extractor:
            f0_result = self.f0_extractor.extract(audio)
            f0_contour = f0_result["f0"]

            # Pool F0 features
            f0_features_raw = f0_result["features"]
            f0_features = torch.from_numpy(f0_features_raw.mean(axis=0)).unsqueeze(0).float()
            f0_features = f0_features.to(self.device)

        # Classify
        logits = self.classifier(mms_features, f0_features)
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)

        tone = predicted.item()
        conf = confidence.item()

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return ToneResult(
            tone=tone,
            confidence=conf,
            tone_name=TONE_NAMES.get(tone, f"Tone {tone + 1}"),
            f0_contour=f0_contour,
            processing_time_ms=processing_time_ms,
            audio_duration_ms=audio_duration_ms,
        )

    def classify_file(self, audio_path: str | Path) -> ToneResult:
        """Classify tone from audio file."""
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000)
        return self.classify(audio)

    def run_interactive(
        self,
        on_result: Optional[Callable[[ToneResult], None]] = None,
        audio_device: Optional[int] = None,
    ):
        """
        Run interactive mode with live microphone input.

        Args:
            on_result: Callback for classification results
            audio_device: Audio device index
        """
        print("\nStarting interactive mode...")
        print("Speak Mandarin syllables into the microphone.")
        print("Press Ctrl+C to stop.\n")

        def handle_utterance(audio: np.ndarray):
            result = self.classify(audio)

            print(f"Tone: {result.tone_name}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Processing: {result.processing_time_ms:.1f}ms")
            print(f"  Audio: {result.audio_duration_ms:.1f}ms")
            print()

            if on_result:
                on_result(result)

        def on_speech_start():
            print("[Listening...]")

        detector = UtteranceDetector(
            on_utterance=handle_utterance,
            on_speech_start=on_speech_start,
            device=audio_device,
        )

        try:
            with detector:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped.")

    def benchmark(self, num_samples: int = 100, audio_duration: float = 0.5) -> dict:
        """
        Benchmark pipeline latency.

        Args:
            num_samples: Number of test samples
            audio_duration: Duration of each test audio in seconds

        Returns:
            Dict with timing statistics
        """
        print(f"Benchmarking with {num_samples} samples...")

        # Generate random test audio
        sr = 16000
        num_samples_audio = int(audio_duration * sr)

        times = []
        for i in range(num_samples):
            audio = np.random.randn(num_samples_audio).astype(np.float32) * 0.1
            result = self.classify(audio)
            times.append(result.processing_time_ms)

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{num_samples}")

        times = np.array(times)

        stats = {
            "mean_ms": float(times.mean()),
            "std_ms": float(times.std()),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }

        print(f"\nLatency statistics:")
        print(f"  Mean: {stats['mean_ms']:.1f}ms")
        print(f"  Std: {stats['std_ms']:.1f}ms")
        print(f"  P50: {stats['p50_ms']:.1f}ms")
        print(f"  P95: {stats['p95_ms']:.1f}ms")
        print(f"  P99: {stats['p99_ms']:.1f}ms")

        return stats


def test_latency():
    """Quick latency test."""
    print("Running latency test...")

    # Create pipeline without checkpoint (uses random weights)
    pipeline = RealtimePipeline(
        classifier_checkpoint=None,
        use_f0=True,
        use_light_encoder=True,
    )

    # Test with different audio lengths
    for duration in [0.3, 0.5, 1.0]:
        sr = 16000
        audio = np.random.randn(int(duration * sr)).astype(np.float32) * 0.1

        result = pipeline.classify(audio)

        print(f"\nAudio duration: {duration:.1f}s")
        print(f"  Processing time: {result.processing_time_ms:.1f}ms")
        print(f"  Predicted tone: {result.tone_name}")
        print(f"  Confidence: {result.confidence:.1%}")

    # Full benchmark
    print("\n" + "=" * 50)
    pipeline.benchmark(num_samples=50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time tone classification")
    parser.add_argument("--test", action="store_true", help="Run latency test")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Classifier checkpoint")
    parser.add_argument("--file", type=str, default=None, help="Classify audio file")
    args = parser.parse_args()

    if args.test:
        test_latency()

    elif args.benchmark:
        pipeline = RealtimePipeline(classifier_checkpoint=args.checkpoint)
        pipeline.benchmark(num_samples=100)

    elif args.interactive:
        pipeline = RealtimePipeline(classifier_checkpoint=args.checkpoint)
        pipeline.run_interactive()

    elif args.file:
        pipeline = RealtimePipeline(classifier_checkpoint=args.checkpoint)
        result = pipeline.classify_file(args.file)
        print(f"Tone: {result.tone_name}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")

    else:
        parser.print_help()
