"""Simple MMS-based ASR for Mandarin Chinese using Hugging Face transformers."""

import os
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
    # Remove spaces, punctuation, keep only Chinese characters
    import re
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", "", text)
    return text


def chars_match(char1: str, char2: str) -> bool:
    """
    Check if two Chinese characters match, considering traditional/simplified variants.

    Uses pinyin comparison as fallback - if two characters have the same pinyin,
    they're considered equivalent (e.g., 謝 and 谢 both = xiè).
    """
    if char1 == char2:
        return True

    # Compare by pinyin (handles traditional/simplified variants)
    try:
        from pypinyin import pinyin, Style
        py1 = pinyin(char1, style=Style.TONE)[0][0] if char1 else ""
        py2 = pinyin(char2, style=Style.TONE)[0][0] if char2 else ""
        return py1 == py2 and py1 != ""
    except Exception:
        return False


def compute_char_matches(transcribed: str, expected: str) -> tuple[list[bool], float]:
    """
    Compute character-level matches between transcribed and expected text.

    Uses a simple alignment: for each expected character, check if it appears
    in the transcription in roughly the right position. Handles traditional/simplified
    variants by comparing pinyin.

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
                trans_char = trans_normalized[trans_idx + offset]
                if chars_match(trans_char, exp_char):
                    found = True
                    trans_idx = trans_idx + offset + 1
                    break

        # Also check if char appears anywhere (for reordering tolerance)
        if not found:
            for trans_char in trans_normalized:
                if chars_match(trans_char, exp_char):
                    found = True
                    break

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
        self.device = os.environ.get("TONE_COACH_DEVICE") or device or ("cuda" if torch.cuda.is_available() else "cpu")

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

        # Try to set up pyctcdecode for better decoding with hotwords
        self._decoder = None
        try:
            from pyctcdecode import build_ctcdecoder

            # Get vocabulary from processor
            vocab = self._processor.tokenizer.get_vocab()
            # Sort by index to get labels in correct order
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            labels = [item[0] for item in sorted_vocab]

            # Build decoder without LM (we'll use hotwords instead)
            self._decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=None,  # No LM, just hotwords
            )
            print("[MMS ASR] pyctcdecode decoder initialized for hotword support.")
        except ImportError:
            print("[MMS ASR] pyctcdecode not available, using greedy decoding.")
        except Exception as e:
            print(f"[MMS ASR] Could not initialize pyctcdecode: {e}")

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

        # Decode - use pyctcdecode with hotwords if available and expected text provided
        if self._decoder is not None and expected_text:
            # Use beam search with hotwords to bias toward expected characters
            # Extract individual characters as hotwords
            hotwords = list(normalize_chinese(expected_text))
            # Also add the full phrase
            if len(expected_text) > 1:
                hotwords.append(expected_text)

            try:
                # Convert logits to numpy for pyctcdecode
                logits_np = logits[0].cpu().numpy()

                # Decode with hotwords
                transcription = self._decoder.decode(
                    logits_np,
                    hotwords=hotwords,
                    hotword_weight=10.0,  # Strong bias toward expected words
                )
            except Exception as e:
                # Fall back to greedy decoding
                print(f"[MMS ASR] Hotword decoding failed: {e}")
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self._processor.batch_decode(predicted_ids)[0]
        else:
            # Standard greedy decoding
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


class WhisperASR:
    """
    Whisper-based ASR for Mandarin Chinese.

    Uses OpenAI's Whisper model via Hugging Face transformers.
    Complements MMS ASR with a different architecture.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: Optional[str] = None,
        lazy_load: bool = True,
    ):
        """
        Initialize Whisper ASR.

        Args:
            model_name: Hugging Face model identifier (whisper-tiny, whisper-base, whisper-small)
            device: Torch device (auto-detected if None)
            lazy_load: If True, load model on first use
        """
        self.model_name = model_name
        self.device = os.environ.get("TONE_COACH_DEVICE") or device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._processor = None
        self._model = None
        self._loaded = False

        if not lazy_load:
            self._load_model()

    @classmethod
    def get_instance(cls, **kwargs) -> "WhisperASR":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    def _load_model(self):
        """Load the Whisper ASR model."""
        if self._loaded:
            return

        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"[Whisper ASR] Loading model {self.model_name}...")

        self._processor = WhisperProcessor.from_pretrained(self.model_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        self._loaded = True
        print("[Whisper ASR] Model loaded.")

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
            expected_text: Optional expected text for matching (also used as prompt hint)

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
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Build generation kwargs
        gen_kwargs = {
            "language": "chinese",
            "task": "transcribe",
            "max_new_tokens": 128,
        }

        # Use expected text as prompt to bias recognition
        # This helps Whisper anticipate the vocabulary we expect
        prompt_ids = None
        if expected_text:
            try:
                # Create a prompt that hints at the expected content
                # Format: "The speaker will say: [expected]"
                # This biases the model toward the expected vocabulary
                prompt = f"Chinese phrase: {expected_text}"
                prompt_ids = self._processor.get_prompt_ids(prompt, return_tensors="pt")
                prompt_ids = prompt_ids.to(self.device)
                gen_kwargs["prompt_ids"] = prompt_ids
            except Exception as e:
                # Fall back to no prompting if it fails
                print(f"[Whisper ASR] Prompt creation failed: {e}")

        # Generate transcription with Chinese language forced
        with torch.no_grad():
            generated_ids = self._model.generate(
                inputs["input_features"],
                **gen_kwargs,
            )

        # Decode - if we used prompt_ids, slice them off
        if prompt_ids is not None:
            # Use the processor's handling of prompt_ids for proper decoding
            transcription = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True, prompt_ids=prompt_ids
            )[0]
        else:
            transcription = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        # Whisper doesn't provide easy confidence scores, estimate from length
        confidence = min(1.0, len(transcription.strip()) / 10) if transcription else 0.0

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

        print("[Whisper ASR] Model unloaded.")


def combine_asr_results(
    result1: Optional[ASRResult],
    result2: Optional[ASRResult],
    expected_text: Optional[str] = None,
) -> ASRResult:
    """
    Combine results from two ASR models using OR logic.

    For each character position, if either model recognized it, mark as recognized.
    Uses the transcription from the model with higher match score.

    Args:
        result1: First ASR result (e.g., MMS)
        result2: Second ASR result (e.g., Whisper)
        expected_text: Expected text for length reference

    Returns:
        Combined ASRResult
    """
    if result1 is None and result2 is None:
        return ASRResult(
            text="",
            pinyin="",
            tones=[],
            confidence=0.0,
            char_matches=[],
            match_score=0.0,
        )

    if result1 is None:
        return result2
    if result2 is None:
        return result1

    # Combine char_matches with OR logic
    n_chars = max(len(result1.char_matches), len(result2.char_matches))
    combined_matches = []
    for i in range(n_chars):
        match1 = result1.char_matches[i] if i < len(result1.char_matches) else False
        match2 = result2.char_matches[i] if i < len(result2.char_matches) else False
        combined_matches.append(match1 or match2)

    combined_score = sum(combined_matches) / len(combined_matches) if combined_matches else 0.0

    # Combine tones with OR logic - prefer non-zero tone
    combined_tones = []
    n_tones = max(len(result1.tones), len(result2.tones))
    for i in range(n_tones):
        tone1 = result1.tones[i] if i < len(result1.tones) else 0
        tone2 = result2.tones[i] if i < len(result2.tones) else 0
        # Prefer the tone from the model that recognized the character
        match1 = result1.char_matches[i] if i < len(result1.char_matches) else False
        match2 = result2.char_matches[i] if i < len(result2.char_matches) else False
        if match1 and not match2:
            combined_tones.append(tone1)
        elif match2 and not match1:
            combined_tones.append(tone2)
        else:
            # Both or neither matched, use first non-zero
            combined_tones.append(tone1 if tone1 else tone2)

    # Use transcription from model with higher match score
    if result1.match_score >= result2.match_score:
        primary = result1
        secondary = result2
    else:
        primary = result2
        secondary = result1

    return ASRResult(
        text=primary.text,
        pinyin=primary.pinyin,
        tones=combined_tones,
        confidence=max(result1.confidence, result2.confidence),
        char_matches=combined_matches,
        match_score=combined_score,
        processing_time_ms=result1.processing_time_ms + result2.processing_time_ms,
    )


# Convenience functions
def get_mms_asr(**kwargs) -> MMSASR:
    """Get singleton MMS ASR instance."""
    return MMSASR.get_instance(**kwargs)


def get_whisper_asr(**kwargs) -> WhisperASR:
    """Get singleton Whisper ASR instance."""
    return WhisperASR.get_instance(**kwargs)


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
