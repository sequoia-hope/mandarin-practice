"""Simple MMS-based ASR for Mandarin Chinese using Hugging Face transformers."""

import time
from dataclasses import dataclass
from typing import Optional
import threading

import numpy as np
import torch


@dataclass
class ASRResult:
    """Result of ASR transcription."""

    text: str  # Transcribed Chinese text
    pinyin: str  # Pinyin of transcribed text
    tones: list[int]  # Tone numbers extracted from pinyin (1-5, 0=unknown)
    confidence: float  # Overall confidence
    char_matches: list[bool]  # Per-character match with expected
    match_score: float  # Fraction of characters matched
    processing_time_ms: float = 0.0


def tone_from_pinyin_syllable(syllable: str) -> int:
    """
    Extract tone number from a pinyin syllable with tone marks.

    Returns tone 1-5, or 0 if no tone mark found (treat as neutral).
    """
    # Tone mark to tone number mapping
    tone_marks = {
        # Tone 1: macron (flat)
        'ā': 1, 'ē': 1, 'ī': 1, 'ō': 1, 'ū': 1, 'ǖ': 1,
        # Tone 2: acute (rising)
        'á': 2, 'é': 2, 'í': 2, 'ó': 2, 'ú': 2, 'ǘ': 2,
        # Tone 3: caron (dipping)
        'ǎ': 3, 'ě': 3, 'ǐ': 3, 'ǒ': 3, 'ǔ': 3, 'ǚ': 3,
        # Tone 4: grave (falling)
        'à': 4, 'è': 4, 'ì': 4, 'ò': 4, 'ù': 4, 'ǜ': 4,
    }

    for char in syllable:
        if char in tone_marks:
            return tone_marks[char]

    # No tone mark = neutral tone (5) or unknown
    return 5


def to_pinyin(text: str) -> str:
    """Convert Chinese text to pinyin with tone marks."""
    try:
        from pypinyin import pinyin, Style
        result = pinyin(text, style=Style.TONE)
        return " ".join(syllable[0] for syllable in result)
    except ImportError:
        return ""
    except Exception:
        return ""


def to_pinyin_with_tones(text: str) -> tuple[str, list[int]]:
    """
    Convert Chinese text to pinyin and extract tone numbers.

    Returns:
        (pinyin_string, list_of_tones)
    """
    try:
        from pypinyin import pinyin, Style
        result = pinyin(text, style=Style.TONE)
        syllables = [syllable[0] for syllable in result]
        tones = [tone_from_pinyin_syllable(s) for s in syllables]
        return " ".join(syllables), tones
    except ImportError:
        return "", []
    except Exception:
        return "", []


def normalize_chinese(text: str) -> str:
    """Normalize Chinese text for comparison."""
    # Remove spaces, punctuation, convert to simplified
    import re

    # Keep only Chinese characters
    text = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return text


def compute_char_matches(transcribed: str, expected: str) -> tuple[list[bool], float]:
    """
    Compute character-level matches between transcribed and expected text.

    Uses a simple alignment: for each expected character, check if it appears
    in the transcription in roughly the right position.

    Returns:
        (list of bools for each expected char, match score 0-1)
    """
    trans_normalized = normalize_chinese(transcribed)
    exp_normalized = normalize_chinese(expected)

    if not exp_normalized:
        return [], 1.0

    matches = []
    trans_idx = 0

    for exp_char in exp_normalized:
        # Look for this character in remaining transcription
        found = False
        search_window = min(3, len(trans_normalized) - trans_idx)  # Look ahead up to 3 chars

        for offset in range(search_window + 1):
            if trans_idx + offset < len(trans_normalized):
                if trans_normalized[trans_idx + offset] == exp_char:
                    found = True
                    trans_idx = trans_idx + offset + 1
                    break

        # Also check if char appears anywhere (for reordering tolerance)
        if not found and exp_char in trans_normalized:
            found = True

        matches.append(found)

    score = sum(matches) / len(matches) if matches else 0.0
    return matches, score


class MMSASR:
    """
    MMS-based ASR for Mandarin Chinese.

    Uses facebook/mms-1b-all model for speech recognition.
    Provides character-level matching against expected text.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "facebook/mms-1b-all",
        device: Optional[str] = None,
        lazy_load: bool = True,
    ):
        """
        Initialize MMS ASR.

        Args:
            model_name: Hugging Face model identifier
            device: Torch device (auto-detected if None)
            lazy_load: If True, load model on first use
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._processor = None
        self._model = None
        self._loaded = False

        if not lazy_load:
            self._load_model()

    @classmethod
    def get_instance(cls, **kwargs) -> "MMSASR":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    def _load_model(self):
        """Load the MMS ASR model."""
        if self._loaded:
            return

        from transformers import Wav2Vec2ForCTC, AutoProcessor

        print(f"[MMS ASR] Loading model {self.model_name}...")

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        # Set language to Mandarin Chinese
        self._processor.tokenizer.set_target_lang("cmn-script_simplified")
        self._model.load_adapter("cmn-script_simplified")

        self._loaded = True
        print("[MMS ASR] Model loaded.")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        expected_text: Optional[str] = None,
    ) -> ASRResult:
        """
        Transcribe audio to Chinese text.

        Args:
            audio: Audio samples (numpy array, mono, 16kHz)
            sample_rate: Sample rate (should be 16000)
            expected_text: Optional expected text for matching

        Returns:
            ASRResult with transcription and match info
        """
        if not self._loaded:
            self._load_model()

        start_time = time.perf_counter()

        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max()

        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Process audio
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Transcribe
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self._processor.batch_decode(predicted_ids)[0]

        # Compute confidence from logits
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        confidence = float(max_probs.mean())

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Compute character matches if expected text provided
        char_matches = []
        match_score = 0.0

        if expected_text:
            char_matches, match_score = compute_char_matches(transcription, expected_text)

        # Convert transcription to pinyin with tones
        transcription_pinyin, transcription_tones = to_pinyin_with_tones(transcription.strip())

        return ASRResult(
            text=transcription.strip(),
            pinyin=transcription_pinyin,
            tones=transcription_tones,
            confidence=confidence,
            char_matches=char_matches,
            match_score=match_score,
            processing_time_ms=processing_time_ms,
        )

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._loaded = False

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[MMS ASR] Model unloaded.")


# Convenience function
def get_mms_asr(**kwargs) -> MMSASR:
    """Get singleton MMS ASR instance."""
    return MMSASR.get_instance(**kwargs)


if __name__ == "__main__":
    # Test the ASR
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Audio file to transcribe")
    parser.add_argument("--expected", type=str, help="Expected text for matching")
    args = parser.parse_args()

    asr = MMSASR(lazy_load=False)

    if args.file:
        import librosa
        audio, sr = librosa.load(args.file, sr=16000)
        result = asr.transcribe(audio, expected_text=args.expected)

        print(f"Transcription: {result.text}")
        print(f"Confidence: {result.confidence:.2%}")
        if args.expected:
            print(f"Match score: {result.match_score:.2%}")
            print(f"Char matches: {result.char_matches}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
    else:
        # Generate test audio
        print("Testing with synthetic audio...")
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)

        result = asr.transcribe(audio, expected_text="你好")
        print(f"Transcription: '{result.text}'")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
