"""Unified tone evaluation service combining encoder, classifier, and aligner."""

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class SyllableResult:
    """Result for a single syllable evaluation."""

    character: str
    pinyin: str
    expected_tone: int  # Underlying/lexical tone
    detected_tone: int
    confidence: float
    correct: bool  # Tone correct
    f0_contour: Optional[np.ndarray] = None
    spoken_tone: Optional[int] = None  # Expected spoken tone (after sandhi)
    sandhi_applied: bool = False  # Whether sandhi changed the expected tone
    char_recognized: Optional[bool] = None  # ASR recognized this character
    asr_tone: Optional[int] = None  # Tone derived from ASR pinyin
    correct_source: Optional[str] = None  # What confirmed correctness: "classifier", "asr", "both", or None


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    syllable_results: list[SyllableResult]
    overall_accuracy: float  # Tone accuracy
    audio: np.ndarray
    duration_ms: int
    status: str = "success"
    error: Optional[str] = None
    # ASR results
    transcription: Optional[str] = None  # What ASR heard (Chinese)
    transcription_pinyin: Optional[str] = None  # Pinyin of what ASR heard
    word_accuracy: Optional[float] = None  # Character match score
    combined_score: Optional[float] = None  # Weighted tone + word score


class Evaluator:
    """
    Unified tone evaluation service.

    Encapsulates:
    - MMS encoder loading (with caching)
    - Tone classifier loading
    - F0 extraction
    - Syllable segmentation (energy-based aligner)
    - Single evaluate() method for complete evaluation
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        use_f0: bool = True,
        use_asr: bool = True,
        asr_backend: str = "both",
        lazy_load: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to classifier checkpoint. If None, auto-detect.
            device: Device to use (auto-detected if None)
            use_f0: Whether to use F0 features
            use_asr: Whether to use ASR for word recognition
            asr_backend: Which ASR to use: "mms", "whisper", or "both"
            lazy_load: If True, load models only when first needed
        """
        # Check for forced CPU mode via environment variable
        env_device = os.environ.get("TONE_COACH_DEVICE")
        if env_device:
            self.device = env_device
        elif device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_f0 = use_f0
        self.use_asr = use_asr
        self.asr_backend = asr_backend

        # Find checkpoint if not specified
        if checkpoint_path is None:
            checkpoint_path = self._find_best_checkpoint()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Components (lazy loaded)
        self._encoder = None
        self._classifier = None
        self._f0_extractor = None
        self._aligner = None
        self._asr_mms = None
        self._asr_whisper = None

        self._loaded = False

        if not lazy_load:
            self._load_components()

    @classmethod
    def get_instance(cls, **kwargs) -> "Evaluator":
        """Get singleton instance of evaluator."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    def _find_best_checkpoint(self) -> Optional[Path]:
        """Find the best checkpoint file based on accuracy in filename."""
        checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"

        if not checkpoint_dir.exists():
            return None

        checkpoints = list(checkpoint_dir.glob("best_model_acc*.pt"))
        if not checkpoints:
            return None

        # Sort by accuracy (extract from filename)
        def get_accuracy(p: Path) -> float:
            try:
                acc_str = p.stem.split("acc")[-1]
                return float(acc_str)
            except ValueError:
                return 0.0

        return max(checkpoints, key=get_accuracy)

    def _load_components(self):
        """Load all model components."""
        if self._loaded:
            return

        print(f"[Evaluator] Loading components on {self.device}...")

        # Load MMS encoder
        from ..features.mms_encoder import MMSEncoder

        print("[Evaluator] Loading MMS encoder...")
        self._encoder = MMSEncoder(device=self.device)

        # Load classifier
        if self.checkpoint_path and self.checkpoint_path.exists():
            from ..model.train import load_model

            print(f"[Evaluator] Loading classifier from {self.checkpoint_path}...")
            self._classifier = load_model(self.checkpoint_path, device=self.device)
        else:
            print("[Evaluator] Warning: No checkpoint found, classifier not loaded")

        # Load F0 extractor
        if self.use_f0:
            from ..features.f0_extract import F0Extractor

            print("[Evaluator] Loading F0 extractor...")
            self._f0_extractor = F0Extractor(method="pyin")

        # Create aligner
        from .aligner import EnergyAligner

        self._aligner = EnergyAligner()

        # Load ASR models for word recognition based on backend setting
        if self.use_asr:
            # Load MMS ASR if requested
            if self.asr_backend in ("mms", "both"):
                try:
                    from .mms_asr import MMSASR

                    print("[Evaluator] Loading MMS ASR...")
                    self._asr_mms = MMSASR(device=self.device, lazy_load=True)
                except Exception as e:
                    print(f"[Evaluator] Warning: MMS ASR not available: {e}")
                    self._asr_mms = None

            # Load Whisper ASR if requested
            if self.asr_backend in ("whisper", "both"):
                try:
                    from .mms_asr import WhisperASR

                    print("[Evaluator] Loading Whisper ASR...")
                    self._asr_whisper = WhisperASR(device=self.device, lazy_load=True)
                except Exception as e:
                    print(f"[Evaluator] Warning: Whisper ASR not available: {e}")
                    self._asr_whisper = None

        self._loaded = True
        print("[Evaluator] Ready!")

    @property
    def encoder(self):
        """Get encoder, loading if needed."""
        if not self._loaded:
            self._load_components()
        return self._encoder

    @property
    def classifier(self):
        """Get classifier, loading if needed."""
        if not self._loaded:
            self._load_components()
        return self._classifier

    @property
    def f0_extractor(self):
        """Get F0 extractor, loading if needed."""
        if not self._loaded:
            self._load_components()
        return self._f0_extractor

    @property
    def aligner(self):
        """Get aligner, loading if needed."""
        if not self._loaded:
            self._load_components()
        return self._aligner

    @property
    def asr_mms(self):
        """Get MMS ASR, loading if needed."""
        if not self._loaded:
            self._load_components()
        return self._asr_mms

    @property
    def asr_whisper(self):
        """Get Whisper ASR, loading if needed."""
        if not self._loaded:
            self._load_components()
        return self._asr_whisper

    def evaluate(
        self,
        audio: np.ndarray,
        expected_pinyin: list[str],
        expected_tones: list[int],
        characters: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate tone pronunciation with sandhi awareness.

        Args:
            audio: Audio samples (1D numpy array, 16kHz)
            expected_pinyin: List of expected pinyin syllables
            expected_tones: List of expected tone numbers (1-5)
            characters: Optional Chinese characters for display

        Returns:
            EvaluationResult with per-syllable and overall results

        Note:
            The evaluator applies tone sandhi rules to determine acceptable
            pronunciations. For example, 你好 [T3, T3] is correctly pronounced
            as [T2, T3] due to third-tone sandhi, and both are accepted.
        """
        if self.classifier is None:
            return EvaluationResult(
                syllable_results=[],
                overall_accuracy=0.0,
                audio=audio,
                duration_ms=int(len(audio) / 16),
                status="error",
                error="Classifier not loaded",
            )

        n_syllables = len(expected_pinyin)

        if n_syllables == 0:
            return EvaluationResult(
                syllable_results=[],
                overall_accuracy=0.0,
                audio=audio,
                duration_ms=int(len(audio) / 16),
                status="error",
                error="No syllables to evaluate",
            )

        # Apply sandhi rules to get spoken tones
        from .sandhi import apply_sandhi, get_acceptable_tones

        sandhi_result = apply_sandhi(expected_tones, expected_pinyin, characters)
        spoken_tones = sandhi_result.spoken_tones

        # Run dual ASR for word recognition (on full audio)
        # Use both MMS and Whisper, combine with OR logic
        asr_result = None
        char_matches = []
        asr_tones = []
        if self.use_asr and characters:
            from .mms_asr import combine_asr_results

            mms_result = None
            whisper_result = None

            # Run MMS ASR
            if self._asr_mms is not None:
                try:
                    mms_result = self._asr_mms.transcribe(audio, expected_text=characters)
                except Exception as e:
                    print(f"[Evaluator] MMS ASR failed: {e}")

            # Run Whisper ASR
            if self._asr_whisper is not None:
                try:
                    whisper_result = self._asr_whisper.transcribe(audio, expected_text=characters)
                except Exception as e:
                    print(f"[Evaluator] Whisper ASR failed: {e}")

            # Combine results (OR logic for char recognition)
            if mms_result is not None or whisper_result is not None:
                asr_result = combine_asr_results(mms_result, whisper_result, characters)
                char_matches = asr_result.char_matches
                asr_tones = asr_result.tones

        # Align audio to syllables
        segments = self.aligner.align(audio, expected_pinyin)

        # Evaluate each syllable
        syllable_results = []

        for i, (start_sec, end_sec) in enumerate(segments):
            # Extract segment
            start_sample = int(start_sec * 16000)
            end_sample = int(end_sec * 16000)
            segment = audio[start_sample:end_sample]

            if len(segment) < 160:  # Less than 10ms
                # Segment too short, skip
                continue

            # Extract MMS features
            features = self.encoder.encode_audio(segment)
            mms_features = features["pooled"]

            # Extract F0 features if enabled
            f0_features = None
            f0_contour = None
            if self.f0_extractor:
                f0_result = self.f0_extractor.extract(segment)
                f0_contour = f0_result.get("f0", np.array([]))

                # Get summary F0 features (mean of features over time)
                f0_features_raw = f0_result.get("features", np.zeros((1, 3)))
                if len(f0_features_raw) > 0:
                    f0_summary = f0_features_raw.mean(axis=0)
                    f0_features = torch.tensor(f0_summary).unsqueeze(0).to(self.device)

            # Classify
            detected_tone, confidence = self._classify_syllable(
                mms_features, f0_features
            )

            # Get character for this syllable
            char = (
                characters[i]
                if characters and i < len(characters)
                else expected_pinyin[i]
            )

            expected_tone = expected_tones[i] if i < len(expected_tones) else 1
            spoken_tone = spoken_tones[i] if i < len(spoken_tones) else expected_tone
            sandhi_applied = (spoken_tone != expected_tone)

            # Get acceptable tones (considering sandhi)
            acceptable = get_acceptable_tones(
                expected_tone=expected_tone,
                position=i,
                all_tones=expected_tones,
                pinyin=expected_pinyin,
                characters=characters,
            )

            # Check if ASR recognized this character
            char_recognized = char_matches[i] if i < len(char_matches) else None

            # Get ASR-derived tone for this syllable
            asr_tone = asr_tones[i] if i < len(asr_tones) else None

            # Tone is correct if ANY of these are true:
            # 1. Tone classifier detected an acceptable tone
            # 2. ASR-derived tone matches acceptable
            # 3. ASR recognized the character (implies pronunciation was close enough)
            # Philosophy: any notion of "correct" makes it correct
            classifier_correct = detected_tone in acceptable
            asr_tone_correct = (
                asr_tone is not None
                and asr_tone in acceptable
            )
            asr_char_correct = char_recognized is True
            is_correct = classifier_correct or asr_tone_correct or asr_char_correct

            # Determine what source(s) confirmed correctness
            sources = []
            if classifier_correct:
                sources.append("classifier")
            if asr_tone_correct:
                sources.append("asr_tone")
            if asr_char_correct:
                sources.append("asr_char")

            if len(sources) == 0:
                correct_source = None
            elif len(sources) == 1:
                correct_source = sources[0]
            else:
                correct_source = "+".join(sources)

            syllable_results.append(
                SyllableResult(
                    character=char,
                    pinyin=expected_pinyin[i],
                    expected_tone=expected_tone,
                    detected_tone=detected_tone,
                    confidence=float(confidence),
                    correct=is_correct,
                    f0_contour=f0_contour,
                    spoken_tone=spoken_tone,
                    sandhi_applied=sandhi_applied,
                    char_recognized=char_recognized,
                    asr_tone=asr_tone,
                    correct_source=correct_source,
                )
            )

        # Calculate overall tone accuracy
        if syllable_results:
            correct_count = sum(1 for r in syllable_results if r.correct)
            overall_accuracy = correct_count / len(syllable_results)
        else:
            overall_accuracy = 0.0

        # Calculate word accuracy from ASR
        transcription = None
        transcription_pinyin = None
        word_accuracy = None
        combined_score = None

        if asr_result is not None:
            transcription = asr_result.text
            transcription_pinyin = asr_result.pinyin
            word_accuracy = asr_result.match_score

            # Combined score: weighted average of tone and word accuracy
            # Tone accuracy is more specific, so weight it higher
            tone_weight = 0.6
            word_weight = 0.4
            combined_score = (
                tone_weight * overall_accuracy + word_weight * word_accuracy
            )

        return EvaluationResult(
            syllable_results=syllable_results,
            overall_accuracy=overall_accuracy,
            audio=audio,
            duration_ms=int(len(audio) / 16),
            status="success",
            transcription=transcription,
            transcription_pinyin=transcription_pinyin,
            word_accuracy=word_accuracy,
            combined_score=combined_score,
        )

    def _classify_syllable(
        self,
        mms_features: torch.Tensor,
        f0_features: Optional[torch.Tensor] = None,
    ) -> tuple[int, float]:
        """
        Classify a single syllable.

        Args:
            mms_features: MMS encoder features (1, 2048)
            f0_features: Optional F0 features (1, 3)

        Returns:
            (tone_number, confidence) where tone_number is 1-5
        """
        with torch.no_grad():
            if f0_features is not None and self.classifier.use_f0:
                logits = self.classifier(mms_features, f0_features)
            else:
                logits = self.classifier(mms_features, None)

            probs = torch.softmax(logits, dim=-1)
            confidence, predicted_class = probs.max(dim=-1)

            # Convert class index (0-4) to tone number (1-5)
            tone_number = predicted_class.item() + 1

            return tone_number, confidence.item()

    def evaluate_from_mic(
        self,
        expected_pinyin: list[str],
        expected_tones: list[int],
        characters: Optional[str] = None,
        timeout: float = 10.0,
    ) -> EvaluationResult:
        """
        Record from microphone and evaluate.

        Args:
            expected_pinyin: Expected pinyin syllables
            expected_tones: Expected tone numbers (1-5)
            characters: Optional Chinese characters
            timeout: Maximum recording time in seconds

        Returns:
            EvaluationResult
        """
        from ..realtime.audio_capture import UtteranceDetector

        utterance_event = threading.Event()
        captured_audio = [None]

        def on_utterance(audio):
            captured_audio[0] = audio
            utterance_event.set()

        def on_speech_start():
            print("  [Listening...]")

        def on_speech_end():
            print("  [Processing...]")

        detector = UtteranceDetector(
            on_utterance=on_utterance,
            on_speech_start=on_speech_start,
            on_speech_end=on_speech_end,
            min_speech_duration_ms=200,
            max_speech_duration_ms=8000,
        )

        print("  [Speak now...]")
        detector.start()

        got_utterance = utterance_event.wait(timeout=timeout)
        detector.stop()

        if not got_utterance or captured_audio[0] is None:
            return EvaluationResult(
                syllable_results=[],
                overall_accuracy=0.0,
                audio=np.zeros(1600, dtype=np.float32),
                duration_ms=0,
                status="timeout",
                error="No speech detected",
            )

        return self.evaluate(
            audio=captured_audio[0],
            expected_pinyin=expected_pinyin,
            expected_tones=expected_tones,
            characters=characters,
        )

    def to_dict(self, result: EvaluationResult) -> dict:
        """Convert EvaluationResult to dictionary for JSON serialization."""
        return {
            "syllable_results": [
                {
                    "character": r.character,
                    "pinyin": r.pinyin,
                    "expected_tone": r.expected_tone,
                    "spoken_tone": r.spoken_tone,
                    "detected_tone": r.detected_tone,
                    "confidence": r.confidence,
                    "correct": r.correct,
                    "sandhi_applied": r.sandhi_applied,
                    "char_recognized": r.char_recognized,
                    "asr_tone": r.asr_tone,
                    "correct_source": r.correct_source,
                }
                for r in result.syllable_results
            ],
            "overall_accuracy": result.overall_accuracy,
            "duration_ms": result.duration_ms,
            "status": result.status,
            "error": result.error,
            "transcription": result.transcription,
            "transcription_pinyin": result.transcription_pinyin,
            "word_accuracy": result.word_accuracy,
            "combined_score": result.combined_score,
        }


# Singleton access
def get_evaluator(**kwargs) -> Evaluator:
    """Get the singleton evaluator instance."""
    return Evaluator.get_instance(**kwargs)


if __name__ == "__main__":
    # Test evaluator
    print("Testing evaluator...")

    evaluator = Evaluator(lazy_load=False)

    # Generate test audio
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Two syllables
    env1 = np.exp(-((t - 0.25) ** 2) / 0.02)
    env2 = np.exp(-((t - 0.75) ** 2) / 0.02)
    audio = ((env1 + env2) * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    result = evaluator.evaluate(
        audio=audio,
        expected_pinyin=["ni", "hao"],
        expected_tones=[3, 3],
        characters="你好",
    )

    print(f"Status: {result.status}")
    print(f"Overall accuracy: {result.overall_accuracy:.1%}")

    for r in result.syllable_results:
        status = "✓" if r.correct else "✗"
        print(
            f"  {status} {r.character} ({r.pinyin}): "
            f"expected T{r.expected_tone}, got T{r.detected_tone} "
            f"(conf: {r.confidence:.2f})"
        )
